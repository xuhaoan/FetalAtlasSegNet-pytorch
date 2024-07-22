import os
import time

import nibabel as nib
import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler

from data_loader import TrainGenerator, get_list
from network.MixAttNet import MixAttNet
from network.UNet import UNet
from network.AtlasSeg import AtlasSeg
from network.SENet import SENet
from network.UNetPP import UNetPP
from utils import check_dir, AvgMeter, dice_score, mae_score
from losses import *
import argparse
import random



def seed_torch(seed=2023):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    random.seed(seed)


def train_batch(model, optimizer, loader, type, patch_size, batch_size):
    model.train()

    image, label = loader.get_item()

    weight = None
    image = Variable(torch.from_numpy(image).cuda())
    label = Variable(torch.from_numpy(label).cuda())

    optimizer.zero_grad()

    predict = model(image)
    if type == 'MixAttNet':
        loss1 = loss_func(predict[0], label, weight)
        loss2 = loss_func(predict[1], label, weight)
        loss3 = loss_func(predict[2], label, weight)
        loss4 = loss_func(predict[3], label, weight)
        loss5 = loss_func(predict[4], label, weight)
        loss6 = loss_func(predict[5], label, weight)
        loss7 = loss_func(predict[6], label, weight)
        loss8 = loss_func(predict[7], label, weight)
        loss9 = loss_func(predict[8], label, weight)
        loss = loss1 + \
               0.8 * loss2 + 0.7 * loss3 + 0.6 * loss4 + 0.5 * loss5 + \
               0.8 * loss6 + 0.7 * loss7 + 0.6 * loss8 + 0.5 * loss9
    else:
        loss = loss_func(predict, label, weight)

    loss.backward()
    optimizer.step()

    return loss.item()


def val(model, val_list, net_type, patch_size):
    model.eval()
    metric_meter = AvgMeter()

    for data_dict in val_list:
        image_path = data_dict['image_path']
        label_path = data_dict['label_path']
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        pre_count = np.zeros_like(image, dtype=np.float32)
        predict = np.zeros_like(image, dtype=np.float32)

        w, h, d = image.shape
        x_list = np.squeeze(np.concatenate((np.arange(0, w - patch_size[0], patch_size[0])[:, np.newaxis],
                                            np.array([w - patch_size[0]])[:, np.newaxis])).astype(np.int))
        y_list = np.squeeze(np.concatenate((np.arange(0, h - patch_size[1], patch_size[1])[:, np.newaxis],
                                            np.array([h - patch_size[1]])[:, np.newaxis])).astype(np.int))
        z_list = np.squeeze(np.concatenate((np.arange(0, d - patch_size[2], patch_size[2])[:, np.newaxis],
                                            np.array([d - patch_size[2]])[:, np.newaxis])).astype(np.int))

        for x in x_list:
            for y in y_list:
                for z in z_list:
                    image_patch = \
                        image[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]].astype(np.float32)[
                            np.newaxis, np.newaxis, ...]
                    if net_type[:8] == 'AtlasSeg':
                        atlas_image_path = data_dict['atlas_image_path']
                        atlas_label_path = data_dict['atlas_label_path']
                        atlas_image = nib.load(atlas_image_path).get_fdata()[x:x + patch_size[0], y:y + patch_size[1],
                                      z:z + patch_size[2]].astype(np.float32)[
                            np.newaxis, np.newaxis, ...]
                        atlas_label = nib.load(atlas_label_path).get_fdata()[x:x + patch_size[0], y:y + patch_size[1],
                                      z:z + patch_size[2]].astype(np.float32)[
                            np.newaxis, np.newaxis, ...]
                        image_patch = np.concatenate([image_patch, atlas_image, atlas_label], axis=1)

                    image_patch_tensor = torch.from_numpy(image_patch).cuda()
                    pre_patch = model(image_patch_tensor).squeeze()
                    predict[x:x + patch_size[0], y:y + patch_size[1],
                    z:z + patch_size[2]] += pre_patch.cpu().data.numpy()
                    pre_count[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] += 1

        predict /= pre_count
        metric_meter.update(dice_score(predict, label))

    return metric_meter.avg


def main(args):
    torch.cuda.set_device(args.local_rank)

    output_path = args.output_path + args.net + '/'
    check_dir(output_path)
    ckpt_path = os.path.join(output_path, "ckpt")
    check_dir(ckpt_path)

    train_list, val_list, test_list = get_list(dir_path=args.data_path, net_type=args.net)

    train_generator = TrainGenerator(train_list,
                                     batch_size=args.batch_size,
                                     patch_size=args.patch_size,
                                     net_type=args.net)
    device = torch.device("cuda", args.local_rank)
    model = (eval(args.net)()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', cooldown=5, patience=5, factor=0.9,
                                                           min_lr=1e-6, verbose=True)

    open(os.path.join(output_path, "train_record.txt"), 'w+')

    loss_meter = AvgMeter()
    start_time = time.time()
    best_metric = 0.

    for iteration in range(1, args.num_iteration + 1):
        train_loss = train_batch(model=model, optimizer=optimizer, loader=train_generator, type=args.net,
                                 patch_size=args.patch_size, batch_size=args.batch_size)
        loss_meter.update(train_loss)

        if iteration % args.pre_fre == 0:
            iteration_time = time.time() - start_time
            info = [iteration, loss_meter.avg, iteration_time]
            print("Iter[{}] | Loss: {:.3f} | Time: {:.2f}".format(*info))
            start_time = time.time()
            loss_meter.reset()

        if iteration % args.val_fre == 0:
            val_dice = val(model, val_list, args.net, args.patch_size)
            scheduler.step(val_dice)
            lr = optimizer.state_dict()["param_groups"][0]["lr"]

            if val_dice > best_metric:
                torch.save(model.state_dict(), os.path.join(ckpt_path, "best_val.pth.gz"))
                best_metric = val_dice
            open(os.path.join(output_path, "train_record.txt"), 'a+').write(
                "{:d} | {:.3f} | {:.3f} | {:.7f}\n".format(iteration, train_loss, val_dice, lr))
            print("Val in Iter[{}] Dice: {:.3f}".format(iteration, val_dice))
        if iteration % args.val_fre == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_path, "train_{}.pth.gz".format(iteration)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--suffix", default='', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_iteration', default=20000, type=int)
    parser.add_argument('--val_fre', default=25, type=int)
    parser.add_argument('--pre_fre', default=25, type=int)
    parser.add_argument('--patch_size', default=[96, 96, 96], type=int)
    parser.add_argument('--data_path',
                        default="/dataset/",
                        type=str)

    parser.add_argument('--mode', default='', type=str)
    parser.add_argument('--output_path', default="/output/",
                        type=str)
    parser.add_argument('--net', default="AtlasSeg", type=str)
    args = parser.parse_args()
    seed_torch()
    main(args)
