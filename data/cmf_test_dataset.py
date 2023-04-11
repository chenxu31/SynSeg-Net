import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
#import random_crop_yh
import pdb

class cmfTestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_A = opt.test_MRI_dir
        self.dir_B = opt.test_CT_dir
        # self.dir_Seg = opt.test_CT_seg_dir

        self.A_paths = opt.imglist_testMRI
        self.B_paths = opt.imglist_testCT

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        """
        if not self.opt.isTrain:
            self.skipcrop = True
        else:
            self.skipcrop = False
        # self.transform = get_transform(opt)

        if self.skipcrop:
            osize = [opt.fineSize, opt.fineSize]
        else:
            osize = [opt.loadSize, opt.loadSize]
        transform_list = []
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        self.transforms_scale = transforms.Compose(transform_list)
        
        transform_list = []
        transform_list.append(transforms.Scale(osize, Image.NEAREST))
        self.transforms_seg_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_crop_yh.randomcrop_yh(opt.fineSize))
        self.transforms_crop = transforms.Compose(transform_list)
        """
        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        self.transforms_normalize = transforms.Compose(transform_list)


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img = Image.open(A_path).convert('L')
        B_img = Image.open(B_path).convert('L')

        A_img = self.transforms_toTensor(A_img)
        B_img = self.transforms_toTensor(B_img)

        A_img = self.transforms_normalize(A_img)
        B_img = self.transforms_normalize(B_img)


        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths':B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'TestDataset'
