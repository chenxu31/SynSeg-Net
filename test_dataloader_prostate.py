import time
import os
import sublist
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import pdb

opt = TrainOptions().parse()
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
