import time
import os
import sublist
import argparse
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import pdb
import torch
import h5py
import numpy as np
import common_metrics
import skimage.io

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

opt = TrainOptions().parse()


# Method = 'ImageOnly'
Method = opt.yh_data_model
TrainOrTest = opt.yh_run_model #'Train' #



MAP_ID_NAME = {
    0: "BG",
    1: "MYO",
    2: "LAC",
    3: "LVC",
    4: "AA"
}
LABEL_COLORS = (
    (255, 0, 0), # red
    (0, 255, 0), # green
    (0, 0, 255), # blue
    (255, 255, 0), # yellow
    (255, 0, 255),
    (0, 255, 255),
)

NUM_CLASSES = 5


def fill_buf(buf, i, img, shape):
    # n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[1]

    sx = int(i%m)*shape[1]
    sy = int(i/m)*shape[0]
    try:
        buf[sy:sy+shape[0], sx:sx+shape[1], :] = img
    except:
        pdb.set_trace()


def visual(X, bgr2rgb=False, need_restore=True):
    assert len(X.shape) == 4
    if need_restore:
        X = restore(X) # np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = int(np.ceil(np.sqrt(X.shape[0])))
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:])

    if bgr2rgb:
        pdb.set_trace()
        buff = buff[:, :, ::-1]

    return buff


def data_restore(im, modality):
    assert modality in ("mr", "ct")

    if modality == "mr":
        mi, ma = -1.8, 4.4
    else:
        mi, ma = -2.8, 3.2

    return ((im - mi) / (ma - mi) * 255.0).astype(np.uint8)


def generate_display_image(image, is_seg=False):
    assert image.ndim == 4

    image = image[:, :, :, :].transpose((2, 3, 1, 0))
    image = image.reshape((image.shape[0], image.shape[1], image.shape[2] * image.shape[3]))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, -1)
    image = visual(image, need_restore=False)
    image = np.expand_dims(image, 0)

    if is_seg:
        new_im = np.tile(image, (1, 1, 1, 3))
        for c in range(1, NUM_CLASSES):
            new_im[:, :, :, 0:1][image == c] = LABEL_COLORS[c - 1][0]
            new_im[:, :, :, 1:2][image == c] = LABEL_COLORS[c - 1][1]
            new_im[:, :, :, 2:3][image == c] = LABEL_COLORS[c - 1][2]
        image = new_im.astype(np.uint8)

    return image


def evaluate(device, opt, net_seg, net_g, test_data, test_label):
    net_seg.eval()
    net_g.eval()
    pred_list = np.zeros_like(test_label)
    fake_list = np.zeros_like(test_data)
    with torch.no_grad():
        dsc_list = np.zeros((test_data.shape[0], opt.output_nc_seg - 1), np.float32)
        assd_list = np.zeros((test_data.shape[0], opt.output_nc_seg - 1), np.float32)
        for i in range(test_data.shape[0]):
            pred = np.zeros([opt.output_nc_seg, test_data.shape[1], test_data.shape[2], test_data.shape[3]], np.float32)
            fake = np.zeros(test_data.shape[1:], np.float32)
            for j in range(test_data.shape[1]):
                patch = test_data[i:i + 1, j:j + 1, :, :]
                patch = torch.tensor(patch, device=device)
                ret = net_seg.forward(patch)
                pred[:, j, :, :] = ret[0].cpu().numpy()
                ret = net_g.forward(patch)
                fake[j, :, :] = ret[0].cpu().numpy()

            pred = pred.argmax(0).astype(np.float32)
            pred_list[i] = pred
            fake_list[i] = fake

            dsc = common_metrics.calc_multi_dice(pred, test_label[i], opt.output_nc_seg)
            assd = common_metrics.calc_multi_assd(pred, test_label[i], opt.output_nc_seg)

            dsc_list[i, :] = dsc
            assd_list[i, :] = assd

    net_seg.train()
    net_g.train()

    return fake_list, pred_list, dsc_list, assd_list


if __name__=='__main__':
    if opt.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    opt.crossentropy_weight = [1,1,1,1,1]
    opt.task = "mmwhs"

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print('#training images = %d' % dataset_size)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    print('#model created')

    f = h5py.File(os.path.join(opt.dataroot, "ct_test.h5"), "r")
    test_data, test_label = np.array(f["data"], np.float32), np.array(f["label"], np.uint8)
    f.close()

    max_dsc = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data)
            model.optimize_parameters()

            """
            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)
            """

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

                train_img = np.concatenate([data_restore(model.real_A.cpu().numpy(), "mr"),
                                            data_restore(model.fake_B.cpu().detach().numpy(), "ct")], 3)
                train_img = generate_display_image(train_img, is_seg=False)
                train_seg = np.concatenate([data["Seg_one"].cpu().numpy(),
                                            np.expand_dims(model.seg_fake_B.cpu().detach().numpy().argmax(1), 1)], 3)
                train_seg = generate_display_image(train_seg, is_seg=True)
                skimage.io.imsave(os.path.join("outputs", "train_img.jpg"), train_img[0, :, :, 0])
                skimage.io.imsave(os.path.join("outputs", "train_seg.jpg"), train_seg[0, :, :, :])

                fake_list, pred_list, dsc, assd = evaluate(device, opt, model.netG_seg, model.netG_B, test_data, test_label)
                cur_dsc = dsc.mean()
                print("step:%d dsc:%f/%f  assd:%f/%f" % (total_steps, cur_dsc, dsc.std(), assd.mean(), assd.std()))

                if cur_dsc > max_dsc:
                    max_dsc = cur_dsc
                    model.save('best')

                """
                img_A = generate_display_image(data_restore(data["A"].cpu().numpy(), "mr"), is_seg=False)
                img_B = generate_display_image(data_restore(data["B"].cpu().numpy(), "ct"), is_seg=False)
                data_seg = generate_display_image(data["Seg_one"].cpu().numpy(), is_seg=True)
                skimage.io.imsave(os.path.join("outputs", "img_A.jpg"), img_A[0, :, :, 0])
                skimage.io.imsave(os.path.join("outputs", "img_B.jpg"), img_B[0, :, :, 0])
                skimage.io.imsave(os.path.join("outputs", "train_seg.jpg"), data_seg[0, :, :, :])
                """

                test_img = np.concatenate([data_restore(test_data[0:1, :, :, :], "ct"),
                                           data_restore(fake_list[0:1, :, :, :], "mr")], 3)
                test_img = generate_display_image(test_img, is_seg=False)
                test_seg = np.concatenate([test_label[0:1, :, :, :], pred_list[0:1, :, :, :]], 3)
                test_seg = generate_display_image(test_seg, is_seg=True)
                skimage.io.imsave(os.path.join("outputs", "test_img.jpg"), test_img[0, :, :, 0])
                skimage.io.imsave(os.path.join("outputs", "test_seg.jpg"), test_seg[0, :, :, :])

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()

    print("max_dsc:%f" % max_dsc)
    """
    os.system("python train_yh.py --name cmf_cyclegan_imgandseg --dataroot %s --batchSize 4 --model test_seg "
              "--pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode cmf_test_seg  --input_nc 1 "
              "--output_nc 1 --checkpoints_dir %s --test_seg_output_dir %s  --display_port 6006" %
              (opt.dataroot, opt.checkpoints_dir, opt.test_seg_output_dir))
    """
