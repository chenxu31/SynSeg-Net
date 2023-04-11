import time
import os
import h5py
import numpy as np
import pdb
import sys
import SimpleITK as sitk
# from models.models import create_model
import torch
from models import networks
import argparse
import common_metrics

MAP_ID_NAME = {
    0: "BG",
    1: "MYO",
    2: "LAC",
    3: "LVC",
    4: "AA"
}
LABEL_COLORS = (
    (255, 0, 0),  # red
    (0, 255, 0),  # green
    (0, 0, 255),  # blue
    (255, 255, 0),  # yellow
    (255, 0, 255),
    (0, 255, 255),
)


def save_nii(img, output_file):
    output_img = sitk.GetImageFromArray(img)
    sitk.WriteImage(output_img, output_file)


def evaluate(args, device):
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    f = h5py.File(os.path.join(args.data_dir, "ct_test.h5"), "r")
    test_data, test_label = np.array(f["data"]), np.array(f["label"])
    f.close()
    f = h5py.File(os.path.join(args.data_dir, "mr_test.h5"), "r")
    mr_test_data, mr_test_label = np.array(f["data"]), np.array(f["label"])
    f.close()

    netG_A = networks.define_G(1, 1, 32, "resnet_9blocks", "instance", False,
                               [args.gpu, ] if args.gpu >= 0 else [], type="A")
    netG_B = networks.define_G(1, 1, 32, "resnet_9blocks", "instance", False,
                               [args.gpu, ] if args.gpu >= 0 else [], type="B")
    netG_seg = networks.define_G(1, args.num_classes, 32, "resnet_9blocks", "instance", False,
                                 [args.gpu, ] if args.gpu >= 0 else [])
    netG_A.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_net_G_A.pth")))
    netG_B.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_net_G_B.pth")))
    netG_seg.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_net_Seg_A.pth")))
    with torch.no_grad():
        pred_list = []
        fake_A_list = []
        fake_B_list = []
        dsc_list = np.zeros((test_data.shape[0], args.num_classes - 1), np.float32)
        assd_list = np.zeros((test_data.shape[0], args.num_classes - 1), np.float32)
        for i in range(test_data.shape[0]):
            pred = np.zeros([args.num_classes, test_data.shape[1], test_data.shape[2], test_data.shape[3]], np.float32)
            fake_A = np.zeros(mr_test_data.shape[1:], np.float32)
            fake_B = np.zeros(test_data.shape[1:], np.float32)
            for j in range(test_data.shape[1]):
                patch = test_data[i:i + 1, j:j + 1, :, :]
                patch = torch.tensor(patch, device=device)
                ret = netG_seg.forward(patch)
                pred[:, j, :, :] = ret[0].cpu().numpy()
                ret = netG_B.forward(patch)
                fake_B[j, :, :] = ret[0].cpu().numpy()

            for j in range(mr_test_data.shape[1]):
                patch_mr = mr_test_data[i:i + 1, j:j + 1, :, :]
                patch_mr = torch.tensor(patch_mr, device=device)
                ret = netG_A.forward(patch_mr)
                fake_A[j, :, :] = ret[0].cpu().numpy()

            fake_A_list.append(fake_A)
            fake_B_list.append(fake_B)
            pred = pred.argmax(0).astype(np.float32)
            pred_list.append(pred)

            dsc = common_metrics.calc_multi_dice(pred, test_label[i], args.num_classes)
            assd = common_metrics.calc_multi_assd(pred, test_label[i], args.num_classes)

            dsc_list[i, :] = dsc
            assd_list[i, :] = assd

    msg = "dsc:%f/%f  assd:%f/%f" % (dsc_list.mean(), dsc_list.std(), assd_list.mean(), assd_list.std())
    for c in range(args.num_classes - 1):
        name = MAP_ID_NAME[c + 1]
        dsc = dsc_list[c]
        assd = assd_list[c]
        msg += "  %s_dsc:%f/%f  %s_assd:%f/%f" % (name, dsc.mean(), dsc.std(), name, assd.mean(), assd.std())

    print(msg)

    if args.output_dir:
        np.save(os.path.join(args.output_dir, "dsc.npy"), dsc_list)
        np.save(os.path.join(args.output_dir, "assd.npy"), assd_list)

        for i, im in enumerate(pred_list):
            save_nii(test_data[i], os.path.join(args.output_dir, "data_%d.nii.gz" % (i + 1)))
            save_nii(test_label[i], os.path.join(args.output_dir, "label_%d.nii.gz" % (i + 1)))
            save_nii(im, os.path.join(args.output_dir, "seg_%d.nii.gz" % (i + 1)))
            save_nii(fake_A_list[i], os.path.join(args.output_dir, "fake_A_%d.nii.gz" % (i + 1)))
            save_nii(fake_B_list[i], os.path.join(args.output_dir, "fake_B_%d.nii.gz" % (i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r'D:\datasets\mmwhs')
    parser.add_argument("--checkpoint_dir", type=str,
                        default=r'D:\training\checkpoints\synseg\mmwhs\cyclegan_imgandseg')
    parser.add_argument("--output_dir", type=str, default=r'D:\training\test_output_synseg\mmwhs')
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    evaluate(args, device)
