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


sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
import common_net_pt as common_net
import common_pelvic_pt as common_pelvic
import common_metrics


def evaluate(args, device):
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test_data_s, test_data_t, test_label_s, test_label_t = common_pelvic.load_test_data(args.data_dir)

    netG_A = networks.define_G(1, 1, 64, "resnet_9blocks", "instance", False,
                               [0, ] if args.gpu >= 0 else [], type="A")
    netG_B = networks.define_G(1, 1, 64, "resnet_9blocks", "instance", False,
                               [0, ] if args.gpu >= 0 else [], type="B")
    netG_seg = networks.define_G(1, common_pelvic.NUM_CLASSES, 64, "resnet_9blocks", "instance", False,
                                 [0, ] if args.gpu >= 0 else [])
    netG_A.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_net_G_A.pth")))
    netG_B.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_net_G_B.pth")))
    netG_seg.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_net_Seg_A.pth")))
    with torch.no_grad():
        pred_list = []
        fake_A_list = []
        fake_B_list = []
        dsc_list = np.zeros((test_data_t.shape[0], common_pelvic.NUM_CLASSES - 1), np.float32)
        assd_list = np.zeros((test_data_t.shape[0], common_pelvic.NUM_CLASSES - 1), np.float32)
        for i in range(test_data_t.shape[0]):
            pred = np.zeros([common_pelvic.NUM_CLASSES, test_data_t.shape[1], test_data_t.shape[2], test_data_t.shape[3]], np.float32)
            fake_A = np.zeros(test_data_s.shape[1:], np.float32)
            fake_B = np.zeros(test_data_t.shape[1:], np.float32)
            for j in range(test_data_t.shape[1]):
                patch = test_data_t[i:i + 1, j:j + 1, :, :]
                patch = torch.tensor(patch, device=device)
                ret = netG_seg.forward(patch)
                pred[:, j, :, :] = ret[0].cpu().numpy()
                ret = netG_B.forward(patch)
                fake_B[j, :, :] = ret[0].cpu().numpy()

            for j in range(test_data_s.shape[1]):
                patch_s = test_data_s[i:i + 1, j:j + 1, :, :]
                patch_s = torch.tensor(patch_s, device=device)
                ret = netG_A.forward(patch_s)
                fake_A[j, :, :] = ret[0].cpu().numpy()

            fake_A_list.append(fake_A)
            fake_B_list.append(fake_B)
            pred = pred.argmax(0).astype(np.float32)
            pred_list.append(pred)

            dsc = common_metrics.calc_multi_dice(pred, test_label_t[i], common_pelvic.NUM_CLASSES)
            assd = common_metrics.calc_multi_assd(pred, test_label_t[i], common_pelvic.NUM_CLASSES)

            dsc_list[i, :] = dsc
            assd_list[i, :] = assd

    msg = "dsc:%f/%f  assd:%f/%f" % (dsc_list.mean(), dsc_list.std(), assd_list.mean(), assd_list.std())
    for c in range(common_pelvic.NUM_CLASSES - 1):
        name = common_pelvic.MAP_ID_NAME[c + 1]
        dsc = dsc_list[c]
        assd = assd_list[c]
        msg += "  %s_dsc:%f/%f  %s_assd:%f/%f" % (name, dsc.mean(), dsc.std(), name, assd.mean(), assd.std())

    print(msg)

    if args.output_dir:
        np.save(os.path.join(args.output_dir, "dsc.npy"), dsc_list)
        np.save(os.path.join(args.output_dir, "assd.npy"), assd_list)

        for i, im in enumerate(pred_list):
            common_pelvic.save_nii(im, os.path.join(args.output_dir, "pred_%d.nii.gz" % i))
            common_pelvic.save_nii(fake_A_list[i], os.path.join(args.output_dir, "syn_st_%d.nii.gz" % i))
            common_pelvic.save_nii(fake_B_list[i], os.path.join(args.output_dir, "syn_ts_%d.nii.gz" % i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r'F:\datasets\pelvic\h5_data')
    parser.add_argument("--checkpoint_dir", type=str, default=r'E:\training\checkpoints\synseg\pelvic\cyclegan_imgandseg')
    parser.add_argument("--output_dir", type=str, default=r'D:\training\test_output\synseg\pelvic')
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    evaluate(args, device)
