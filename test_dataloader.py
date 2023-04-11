import time
import os
import sublist
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import pdb

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

opt = TrainOptions().parse()



# Method = 'ImageOnly'
Method = opt.yh_data_model


train_MRI_list = r"D:\datasets\mri_seg_disentangle\synseg\train_mri.txt"
train_CT_list = r"D:\datasets\mri_seg_disentangle\synseg\train_ct.txt"

train_MRI_dir = r"D:\datasets\mri_seg_disentangle\synseg\train\mri"
train_CT_dir = r"D:\datasets\mri_seg_disentangle\synseg\train\ct"
train_SEG_dir = r"D:\datasets\mri_seg_disentangle\synseg\train\seg"
test_MRI_dir = r"D:\datasets\mri_seg_disentangle\synseg\test\mri"
test_SEG_dir = r"D:\datasets\mri_seg_disentangle\synseg\test\seg"


TrainOrTest = opt.yh_run_model #'Train' #

imglist_MRI = sublist.dir2list(train_MRI_dir, train_MRI_list)
imglist_CT = sublist.dir2list(train_CT_dir, train_CT_list)

opt.MRI_dir = train_MRI_dir
opt.CT_dir = train_CT_dir
opt.SEG_dir = train_SEG_dir
opt.imglist_MRI = imglist_MRI
opt.imglist_CT = imglist_CT

opt.crossentropy_weight = [1,1,10,10,1,10,1]

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

print('#training images = %d' % dataset_size)

from PIL import Image
import numpy

if __name__=='__main__':
    for i, data in enumerate(dataset):
        im_a = ((data["A"][0, 0].numpy() + 1) * 127.5).astype(numpy.uint8)
        im_b = ((data["B"][0, 0].numpy() + 1) * 127.5).astype(numpy.uint8)
        im_seg = (data["Seg_one"][0, 0].numpy() * 255).astype(numpy.uint8)
        Image.fromarray(im_a).save("ima.jpg")
        Image.fromarray(im_b).save("imb.jpg")
        Image.fromarray(im_seg).save("ims.jpg")
        pdb.set_trace()
