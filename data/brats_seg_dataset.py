import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
import data.random_crop_yh as random_crop_yh
import pdb
import h5py
import numpy
import platform
import sys


if platform.system() == 'Windows':
    NUM_WORKERS = 0
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    NUM_WORKERS = 4
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_brats_goat as common_brats


NUM_CLASSES = 2


class bratsSegDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.data_A = None
        self.data_B = None
        self.data_Seg = None

        f_A = h5py.File(os.path.join(opt.dataroot, "train_%s.h5" % opt.src_modality), "r")
        f_B = h5py.File(os.path.join(opt.dataroot, "train_%s.h5" % opt.dst_modality), "r")
        self.half_subjects = f_A["data"].shape[0] // 2
        self.channels = f_A["data"].shape[1]
        self.A_size = self.half_subjects * self.channels
        self.B_size = self.half_subjects * self.channels
        f_A.close()
        f_B.close()
        self.skipcrop = True
        # self.transform = get_transform(opt)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        #transform_list = []
        #transform_list.append(transforms.Normalize((0.,), (0.,)))
        #self.transforms_normalize = transforms.Compose(transform_list)

    def data_normalize(self, im):
        return common_brats.data_normalize(im)

    def __getitem__(self, index):
        #index_A = random.randint(0, self.B_size - 1) # index #% self.A_size
        #index_B = random.randint(0, self.B_size - 1)

        if self.data_A is None:
            self.data_A = h5py.File(os.path.join(self.opt.dataroot, "train_%s.h5" % self.opt.src_modality), "r")["data"]
        if self.data_B is None:
            self.data_B = h5py.File(os.path.join(self.opt.dataroot, "train_%s.h5" % self.opt.dst_modality), "r")["data"]
        if self.data_Seg is None:
            self.data_Seg = h5py.File(os.path.join(self.opt.dataroot, "train_seg.h5"), "r")["data"]

        A_id1 = random.randint(0, self.half_subjects - 1)
        B_id1 = random.randint(self.half_subjects, self.half_subjects * 2 - 1)
        A_id2 = random.randint(0, self.channels - 1)
        B_id2 = random.randint(0, self.channels - 1)

        A_img = numpy.array(self.data_A[A_id1, A_id2:A_id2 + 1, :, :], numpy.float32).transpose((1, 2, 0))
        B_img = numpy.array(self.data_B[B_id1, B_id2:B_id2 + 1, :, :], numpy.float32).transpose((1, 2, 0))
        Seg_img = numpy.array(self.data_Seg[A_id1, A_id2:A_id2 + 1, :, :], numpy.int64).transpose((1, 2, 0))

        A_img = self.data_normalize(A_img)
        B_img = self.data_normalize(B_img)

        A_img = self.transforms_toTensor(A_img)
        B_img = self.transforms_toTensor(B_img)
        Seg_img = self.transforms_toTensor(Seg_img)

        #A_img = self.transforms_normalize(A_img)
        #B_img = self.transforms_normalize(B_img)

        #Seg_img[Seg_img == 255] = 1

        Seg_img[Seg_img > 0] = 1
        Seg_imgs = torch.Tensor(NUM_CLASSES, Seg_img.shape[1], Seg_img.shape[2])
        for i in range(NUM_CLASSES):
            Seg_imgs[i, :, :] = Seg_img == i

        return {'A': A_img, 'B': B_img, 'Seg': Seg_imgs, 'Seg_one': Seg_img,
                'A_paths': "A_paths", 'B_paths': "B_path", 'Seg_paths':"Seg_path"}


    def __len__(self):
        return self.A_size #max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
