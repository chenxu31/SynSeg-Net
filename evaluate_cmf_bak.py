import time
import os
import cv2
import h5py
import numpy
import pdb
import sys
import SimpleITK as sitk


sys.path.append(r"D:\Dropbox\sourcecode\python\util")
import common_metrics


VIEW = "sagittal"
test_subject_ids = [1, 2, 3, 4, 5, 6, 7, 8]
data_dir = r"D:\datasets\mri_seg_disentangle"
result_dir = os.path.join(r"D:\training\test_synseg_" + VIEW, "img_fake_only", "test", "mri")


def save_nii(img, output_file):
    output_img = img.transpose((0, 2, 1))
    output_img = sitk.GetImageFromArray(output_img)
    sitk.WriteImage(output_img, output_file)


def pad(image, view, pad_value=-1):
    if view == "coronal":
        image = numpy.pad(image, ((0, 0), (3, 3), (2, 2), (0, 0)), 'constant', constant_values=pad_value)
    elif view == "axial":
        image = numpy.pad(image.transpose((0, 3, 2, 1)), ((0, 0), (1, 1), (2, 2), (0, 0)), 'constant', constant_values=pad_value)
    elif view == "sagittal":
        image = numpy.pad(image.transpose((0, 1, 3, 2))[:, ::-1, :, :], ((0, 0), (3, 3), (1, 1), (0, 0)),
                          'constant', constant_values=pad_value)
    else:
        assert 0

    return image


def calc_dice(im1, im2, tid=1.):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 == tid  # make it boolean
    im2 = im2 == tid  # make it boolean
    im1 = numpy.asarray(im1).astype(numpy.bool)
    im2 = numpy.asarray(im2).astype(numpy.bool)

    s = im1.sum() + im2.sum()
    if s == 0:
        return 1

    # Compute Dice coefficient
    intersection = numpy.logical_and(im1, im2)
    dsc = 2. * intersection.sum() / s
    return dsc


def evaluate():
    data = h5py.File(os.path.join(data_dir, "paired_mri_ct.h5"), "r")
    #seg = numpy.pad(numpy.array(data["seg"]), ((0, 0), (3, 3), (2, 2), (0, 0)), 'constant', constant_values=0)
    seg = pad(numpy.array(data["seg"]), view=VIEW, pad_value=0)

    dsc_list = []
    assd_list = []
    for i in range(seg.shape[0]):
        seg_fake = numpy.zeros_like(seg[i])
        for j in range(seg.shape[3]):
            im = cv2.imread(os.path.join(result_dir, "%d_%d_fake_B.png" % (i, j)), cv2.IMREAD_GRAYSCALE)
            seg_fake[:, :, j] = im

            """
            if i==5 and j == 80:
                cv2.imwrite("test_fake.jpg", im)
                cv2.imwrite("test_real.jpg", (seg[i, :, :, j] * 255).astype(numpy.uint8))
                pdb.set_trace()
            """

        seg_fake[seg_fake == 255] = 1
        save_nii(seg_fake, "seg_fake.nii.gz")

        dsc_list.append(calc_dice(seg[i], seg_fake))
        assd_list.append(common_metrics.assd(seg_fake, seg[i]))

    dsc_list = numpy.array(dsc_list)
    assd_list = numpy.array(assd_list)
    print("dsc:%f/%f  assd:%f/%f" % (dsc_list.mean(), dsc_list.std(), assd_list.mean(), assd_list.std()))


if __name__=='__main__':
    evaluate()
