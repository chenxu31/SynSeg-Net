import time
import os
import h5py
import numpy as np
import pdb
import sys
import SimpleITK as sitk
#from models.models import create_model
import torch
from models import networks
import argparse
import common_metrics


def pad(image, pad_value=-1):
    if image.ndim == 4:
        image = np.pad(image, ((0, 0), (0, 0), (2, 2), (1, 1)), 'constant', constant_values=pad_value).transpose((0, 3, 2, 1))
    else:
        image = np.pad(image, ((0, 0), (2, 2), (1, 1)), 'constant', constant_values=pad_value).transpose((2, 1, 0))

    return image


def unpad(image):
    if image.ndim == 4:
        image = image[:, 1:-1, 2:-2, :].transpose((0, 3, 2, 1))
    else:
        image = image[1:-1, 2:-2, :].transpose((2, 1, 0))

    return image


def save_nii(img, output_file):
    output_img = img.transpose((0, 2, 1))
    output_img = sitk.GetImageFromArray(output_img)
    sitk.WriteImage(output_img, output_file)


def evaluate(args, device):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    paired_data = h5py.File(os.path.join(args.data_dir, "paired_mri_ct.h5"), "r")
    paired_mri = pad(np.array(paired_data["mri"]))
    paired_ct = pad(np.array(paired_data["ct"]))
    paired_seg = pad(np.array(paired_data["seg"]), 0)
    paired_data.close()

    netG_A = networks.define_G(1, 1, 64, "resnet_9blocks", "instance", False,
                               [args.gpu,] if args.gpu >= 0 else [])
    netG_B = networks.define_G(1, 1, 64, "resnet_9blocks", "instance", False,
                               [args.gpu,] if args.gpu >= 0 else [])
    netG_seg = networks.define_G(1, 2, 64, "resnet_9blocks", "instance", False,
                                 [args.gpu,] if args.gpu >= 0 else [])
    netG_A.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "latest_net_G_A.pth")))
    netG_B.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "latest_net_G_B.pth")))
    netG_seg.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "latest_net_Seg_A.pth")))
    with torch.no_grad():
        dsc_list = np.zeros((paired_mri.shape[0], 2), np.float32)
        assd_list = np.zeros((paired_mri.shape[0], 2), np.float32)
        fake_dsc_list = np.zeros((paired_mri.shape[0], 2), np.float32)
        fake_assd_list = np.zeros((paired_mri.shape[0], 2), np.float32)
        for i in range(paired_mri.shape[0]):
            pred = np.zeros([2, paired_mri.shape[1], paired_mri.shape[2], paired_mri.shape[3]], np.float32)
            fake_pred = np.zeros([2, paired_mri.shape[1], paired_mri.shape[2], paired_mri.shape[3]], np.float32)
            mri_fake = np.zeros(paired_mri.shape[1:], np.float32)
            ct_fake = np.zeros(paired_mri.shape[1:], np.float32)
            for j in range(paired_mri.shape[3]):
                mri_patch = paired_mri[i:i + 1, :, :, j:j + 1].transpose((0, 3, 1, 2))
                ct_patch = paired_ct[i:i + 1, :, :, j:j + 1].transpose((0, 3, 1, 2))
                mri_patch = torch.tensor(mri_patch, device=device)
                ct_patch = torch.tensor(ct_patch, device=device)

                ret = netG_B.forward(mri_patch)
                ct_fake[:, :, j] = ret[0, 0, :, :].cpu().numpy()

                ret = netG_seg.forward(mri_patch)
                pred[:, :, :, j] = ret[0].cpu().numpy()

                fake_patch = netG_A.forward(ct_patch)
                mri_fake[:, :, j] = fake_patch[0, 0, :, :].cpu().numpy()

                ret = netG_seg.forward(fake_patch)
                fake_pred[:, :, :, j] = ret[0].cpu().numpy()

            pred = pred.argmax(0).astype(np.float32)
            fake_pred = fake_pred.argmax(0).astype(np.float32)

            mri_fake = unpad(mri_fake)
            ct_fake = unpad(ct_fake)
            pred = unpad(pred)
            fake_pred = unpad(fake_pred)
            label = unpad(paired_seg[i])

            if args.output_dir:
                save_nii(mri_fake, os.path.join(args.output_dir, "mri_%d.nii.gz" % (i + 1)))
                save_nii(ct_fake, os.path.join(args.output_dir, "ct_%d.nii.gz" % (i + 1)))
                save_nii(pred, os.path.join(args.output_dir, "mri_seg_%d.nii.gz" % (i + 1)))
                save_nii(fake_pred, os.path.join(args.output_dir, "mri_fake_seg_%d.nii.gz" % (i + 1)))

            dsc = common_metrics.dc(pred, label)
            assd = common_metrics.assd(pred, label)

            fake_dsc = common_metrics.dc(fake_pred, label)
            fake_assd = common_metrics.assd(fake_pred, label)

            dsc_list[i, :] = dsc
            assd_list[i, :] = assd
            fake_dsc_list[i, :] = fake_dsc
            fake_assd_list[i, :] = fake_assd

    msg = "dsc:%f/%f  assd:%f/%f  fake_dsc:%f/%f  fake_assd:%f/%f" % \
          (dsc_list.mean(), dsc_list.std(), assd_list.mean(), assd_list.std(),
           fake_dsc_list.mean(), fake_dsc_list.std(), fake_assd_list.mean(), fake_assd_list.std())

    print(msg)

    np.save(os.path.join(args.output_dir, "dsc.npy"), dsc_list)
    np.save(os.path.join(args.output_dir, "assd.npy"), assd_list)
    np.save(os.path.join(args.output_dir, "fake_dsc.npy"), fake_dsc_list)
    np.save(os.path.join(args.output_dir, "fake_assd.npy"), fake_assd_list)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r'D:\datasets\mri_seg_disentangle')
    parser.add_argument("--checkpoint_dir", type=str, default=r'D:\training\checkpoints\synseg_axial\cmf_cyclegan_imgandseg')
    parser.add_argument("--output_dir", type=str, default=r'D:\training\test_output_synseg\cmf_axial')
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")
        
    evaluate(args, device)
