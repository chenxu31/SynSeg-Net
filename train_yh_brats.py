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
import skimage.io
import platform
import sys

if platform.system() == "Windows":
    sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
    sys.path.append("/home/chenxu31/sourcecode/python/util")
import common_net_pt as common_net
import common_brats_goat as common_brats
import common_metrics

opt = TrainOptions().parse()


# Method = 'ImageOnly'
Method = opt.yh_data_model
TrainOrTest = opt.yh_run_model #'Train' #


def produce_result(device, data, net_seg, num_classes, is_seg=True):
    if is_seg:
        pred = np.zeros([num_classes, data.shape[0], data.shape[1], data.shape[2]], np.float32)
    else:
        pred = np.zeros(data.shape, np.float32)
    for i in range(data.shape[0]):
        patch = np.expand_dims(data[i:i + 1, :, :], 0)
        patch = torch.tensor(patch, device=device)
        ret = net_seg.forward(patch)
        if is_seg:
            pred[:, i, :, :] = ret[0].cpu().numpy()
        else:
            pred[i, :, :] = ret[0].cpu().numpy()

    return pred


def evaluate(device, opt, net_seg, val_data_t, val_label_t):
    net_seg.eval()

    val_t_pred_list = []
    val_t_dsc_list = np.zeros((val_data_t.shape[0], ), np.float32)
    val_t_assd_list = np.zeros((val_data_t.shape[0], ), np.float32)
    with torch.no_grad():
        for i in range(val_data_t.shape[0]):
            t_pred = produce_result(device, val_data_t[i], net_seg, common_brats.NUM_CLASSES)
            t_pred = t_pred.argmax(0).astype(np.float32)
            t_dsc = common_metrics.dc(t_pred, val_label_t[i])
            t_assd = common_metrics.assd(t_pred, val_label_t[i])
            val_t_dsc_list[i] = t_dsc
            val_t_assd_list[i] = t_assd
            val_t_pred_list.append(t_pred)

    net_seg.train()

    return val_t_dsc_list, val_t_assd_list, val_t_pred_list


if __name__=='__main__':
    """
    if opt.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")
    """
    device = opt.device

    opt.crossentropy_weight = [1, 1]
    opt.task = "brats"

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print('#training images = %d' % dataset_size)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    print('#model created')

    val_data_t, val_label_t = common_brats.load_test_data(opt.dataroot, "val", (opt.dst_modality, "seg"))

    max_dsc = 0
    best_test_dsc = 0
    best_test_assd = 0
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

                train_pred = model.seg_fake_B.cpu().detach().numpy().argmax(1)
                train_dsc = common_metrics.dc(train_pred, data["Seg_one"].cpu().detach().numpy())
                train_dsc = np.array(train_dsc)
                msg = "step:%d train_dsc:%f/%f" % (total_steps, train_dsc.mean(), train_dsc.std())

                train_img = np.concatenate([model.real_A.cpu().numpy(), model.fake_B.cpu().detach().numpy()], 3)
                train_img = common_brats.generate_display_image(train_img, is_seg=False)
                train_seg = np.concatenate([data["Seg_one"].cpu().numpy(),
                                            np.expand_dims(train_pred, 1)], 3)
                train_seg = common_brats.generate_display_image(train_seg, is_seg=True)
                skimage.io.imsave(os.path.join("outputs", "train_img.jpg"), train_img)
                skimage.io.imsave(os.path.join("outputs", "train_seg.jpg"), train_seg)

                val_t_dsc, val_t_assd, val_t_pred_list = evaluate(device, opt, model.netG_seg, val_data_t, val_label_t)

                cur_dsc = val_t_dsc.mean()
                msg = "val_dsc:%f/%f  val_assd:%f/%f" % (cur_dsc, val_t_dsc.std(), val_t_assd.mean(), val_t_assd.std())
                print(msg)

                if cur_dsc > max_dsc:
                    max_dsc = cur_dsc
                    best_test_dsc = test_t_dsc
                    best_test_assd = test_t_assd
                    model.save('best')

                """
                img_A = generate_display_image(data_restore(data["A"].cpu().numpy(), "mr"), is_seg=False)
                img_B = generate_display_image(data_restore(data["B"].cpu().numpy(), "ct"), is_seg=False)
                data_seg = generate_display_image(data["Seg_one"].cpu().numpy(), is_seg=True)
                skimage.io.imsave(os.path.join("outputs", "img_A.jpg"), img_A[0, :, :, 0])
                skimage.io.imsave(os.path.join("outputs", "img_B.jpg"), img_B[0, :, :, 0])
                skimage.io.imsave(os.path.join("outputs", "train_seg.jpg"), data_seg[0, :, :, :])
                """

                test_seg = np.expand_dims(np.concatenate([val_label_t[0], val_t_pred_list[0]], 2), 0)
                test_seg = common_brats.generate_display_image(test_seg, is_seg=True)
                skimage.io.imsave(os.path.join("outputs", "test_seg.jpg"), test_seg)

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

    print("Best_dsc:%f/%f" % (best_test_dsc.mean(), best_test_dsc.std()))
    print("Best_assd:%f/%f" % (best_test_assd.mean(), best_test_assd.std()))
