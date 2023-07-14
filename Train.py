import os
import time

import nibabel as nib
import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler

from DataOp import TrainGenerator, get_list
from Network.MixAttNet import MixAttNet
from Network.UNet import UNet
from Network.AtlasSeg import AtlasSegNet
from Network.AtlasSegJoint import AtlasSegJointNet
from Network.AtlasSegSTN import AtlasSegSTNNet

from Network.SENet import SENet
from Network.GASegNet import GASegNet
from Utils import check_dir, AvgMeter, dice_score, mae_score
from losses import *
import argparse


def seed_torch(seed=42):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def adjust_lr(optimizer, iteration, num_iteration):
    """
    we decay the learning rate by a factor of 0.1 in 1/2 and 3/4 of whole training process
    """
    if iteration == num_iteration // 2:
        lr = 1e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif iteration == num_iteration // 4 * 3:
        lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass


def train_batch(model, optimizer, loader, scaler, type, patch_size, batch_size):
    model.train()

    if type[:2] == 'GA':
        image, label, GA = loader.get_item()
        GA = Variable(torch.from_numpy(GA).cuda())
    else:
        image, label = loader.get_item()

    # if np.where(label == 1)[0].shape[0] == 0:
    #    weight = 1
    # else:
    #    weight = batch_size * patch_size * patch_size * patch_size / np.where(label == 1)[0].shape[0]
    weight = None

    image = Variable(torch.from_numpy(image).cuda())
    label = Variable(torch.from_numpy(label).cuda())

    optimizer.zero_grad()

    with autocast():

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
        elif type[:2] == 'GA':
            loss = loss_mae_sync(predict, label, GA, weight)
        else:
            loss = loss_func(predict, label, weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    #     # loss.backward()
    #     # optimizer.step()

    # predict = model(image)
    #
    # if type == 'MixAttNet':
    #     loss1 = loss_func(predict[0], label, weight)
    #     loss2 = loss_func(predict[1], label, weight)
    #     loss3 = loss_func(predict[2], label, weight)
    #     loss4 = loss_func(predict[3], label, weight)
    #     loss5 = loss_func(predict[4], label, weight)
    #     loss6 = loss_func(predict[5], label, weight)
    #     loss7 = loss_func(predict[6], label, weight)
    #     loss8 = loss_func(predict[7], label, weight)
    #     loss9 = loss_func(predict[8], label, weight)
    #     loss = loss1 + \
    #            0.8 * loss2 + 0.7 * loss3 + 0.6 * loss4 + 0.5 * loss5 + \
    #            0.8 * loss6 + 0.7 * loss7 + 0.6 * loss8 + 0.5 * loss9
    # elif type[:2] == 'GA':
    #     loss = loss_mae_sync(predict, label, GA, weight)
    # else:
    #     loss = loss_func(predict, label, weight)
    # loss.backward()
    # optimizer.step()

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
        x_list = np.squeeze(np.concatenate((np.arange(0, w - patch_size[0], patch_size[0] // 4)[:, np.newaxis],
                                            np.array([w - patch_size[0]])[:, np.newaxis])).astype(np.int))
        y_list = np.squeeze(np.concatenate((np.arange(0, h - patch_size[1], patch_size[1] // 4)[:, np.newaxis],
                                            np.array([h - patch_size[1]])[:, np.newaxis])).astype(np.int))
        z_list = np.squeeze(np.concatenate((np.arange(0, d - patch_size[2], patch_size[2] // 4)[:, np.newaxis],
                                            np.array([d - patch_size[2]])[:, np.newaxis])).astype(np.int))

        for x in x_list:
            for y in y_list:
                for z in z_list:
                    image_patch = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]].astype(np.float32)[np.newaxis, np.newaxis, ...]
                    if net_type[:8] == 'AtlasSeg':
                        atlas_image_path = data_dict['atlas_image_path']
                        atlas_label_path = data_dict['atlas_label_path']
                        atlas_image = nib.load(atlas_image_path).get_fdata()[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]].astype(np.float32)[
                            np.newaxis, np.newaxis, ...]
                        atlas_label = nib.load(atlas_label_path).get_fdata()[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]].astype(np.float32)[
                            np.newaxis, np.newaxis, ...]
                        image_patch = np.concatenate([image_patch, atlas_image, atlas_label], axis=1)
                    elif net_type[:2] == 'GA':
                        GA_path = data_dict['GA_path']
                        with open(GA_path, "r") as f:
                            GA = f.read()

                    if net_type[:2] == 'GA':
                        image_patch_tensor = torch.from_numpy(image_patch).cuda()
                        pre_patch = model(image_patch_tensor)
                        predict[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] += pre_patch[0].squeeze().cpu().data.numpy()
                        pre_count[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] += 1
                        pred_GA = pre_patch[1].squeeze()
                        pred_GA = pred_GA.cpu().data.numpy()

                    else:
                        image_patch_tensor = torch.from_numpy(image_patch).cuda()
                        pre_patch = model(image_patch_tensor).squeeze()
                        predict[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += pre_patch.cpu().data.numpy()
                        pre_count[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += 1
        predict /= pre_count
        if net_type[:2] == 'GA':
            metric_meter.update(dice_score(predict, label) - 0.1 * mae_score(pred_GA, GA))
        else:
            metric_meter.update(dice_score(predict, label))

    return metric_meter.avg


def main(args):
    torch.cuda.set_device(args.local_rank)

    torch.distributed.init_process_group(backend='nccl')

    output_path = args.output_path + args.net + args.suffix + '/'
    check_dir(output_path)
    ckpt_path = os.path.join(output_path, "ckpt")
    check_dir(ckpt_path)

    train_list, val_list, test_list = get_list(dir_path=args.data_path + args.mode + '/', net_type=args.net)

    train_generator = TrainGenerator(train_list,
                                     batch_size=args.batch_size,
                                     patch_size=args.patch_size,
                                     net_type=args.net)
    device = torch.device("cuda", args.local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(eval(args.net)()).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, min_lr=5e-6,
                                                           verbose=True)
    scaler = GradScaler()
    open(os.path.join(output_path, "train_record.txt"), 'w+')

    loss_meter = AvgMeter()
    start_time = time.time()
    best_metric = 0.

    for iteration in range(1, args.num_iteration + 1):
        # adjust_lr(optimizer, iteration, args.num_iteration)
        train_loss = train_batch(model=model, optimizer=optimizer, loader=train_generator, scaler=scaler, type=args.net,
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
            if val_dice > best_metric:
                if torch.distributed.get_rank() == 0:
                    torch.save(model.module.state_dict(), os.path.join(ckpt_path, "best_val.pth.gz"))
                best_metric = val_dice
            open(os.path.join(args.output_path, "train_record.txt"), 'a+').write(
                "{:.3f} | {:.3f}\n".format(train_loss, val_dice))
            print("Val in Iter[{}] Dice: {:.3f}".format(iteration, val_dice))
        if iteration % 100 == 0:
            if torch.distributed.get_rank() == 0:
                torch.save(model.module.state_dict(), os.path.join(ckpt_path, "train_{}.pth.gz".format(iteration)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--suffix", default='', type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--num_iteration', default=10000, type=int)
    parser.add_argument('--val_fre', default=100, type=int)
    parser.add_argument('--pre_fre', default=20, type=int)
    parser.add_argument('--patch_size', default=[96, 96, 96], type=int)
    parser.add_argument('--data_path',
                        default="/home/xuhaoan/Program/fetal_segmentation/GA-UNet/DouNet/dataset/",
                        type=str)

    parser.add_argument('--mode', default='resized', type=str)
    parser.add_argument('--output_path', default="/home/xuhaoan/Program/fetal_segmentation/GA-UNet/DouNet/output/",
                        type=str)
    parser.add_argument('--net', default="MixAttNet", type=str)
    args = parser.parse_args()
    # seed_torch()
    main(args)
