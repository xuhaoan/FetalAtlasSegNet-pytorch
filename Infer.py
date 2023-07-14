import os
import time

import nibabel as nib
import numpy as np

import torch
import pandas as pd
from Network.MixAttNet import MixAttNet
from Network.UNet import UNet
from Network.AtlasSeg import AtlasSegNet

from Network.AtlasSegJoint import AtlasSegJointNet
from Network.AtlasSegSTN import AtlasSegSTNNet

from Network.SENet import SENet
from Network.GASegNet import GASegNet
from Utils import AvgMeter, check_dir
from DataOp import get_list
from medpy.metric.binary import dc
import argparse

torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='train')

parser.add_argument('--net', default="MixAttNet", type=str)
parser.add_argument('--suffix', default='', type=str)

parser.add_argument('--mode', default='resized', type=str)
parser.add_argument('--min', default=None, type=int)
parser.add_argument('--max', default=None, type=int)
args = parser.parse_args()

output_path = '/home/xuhaoan/Program/fetal_segmentation/GA-UNet/DouNet/output/' + args.net + args.suffix + '/'
path = '/home/xuhaoan/Program/fetal_segmentation/GA-UNet/DouNet/dataset/'+args.mode+'/'
_, _, test_list = get_list(dir_path=path,
                           net_type=args.net)
net = eval(args.net)().cuda()
net.load_state_dict(torch.load(output_path + '/ckpt/best_val.pth.gz'))

patch_size = [96,96,96]
spacing = 1

save_path = os.path.join(output_path, 'save_data')
check_dir(save_path)

net.eval()

test_meter = AvgMeter()
DICE = np.zeros([32, 100])
Dice = np.zeros([32, 1])
dice_sum = 0
mae_sum = 0
for idx, data_dict in enumerate(test_list):
    image_path = data_dict['image_path']
    label_path = data_dict['label_path']
    name = image_path.split('/')[-1]
    image = nib.load(image_path).get_fdata()
    y_gt = nib.load(label_path).get_fdata()

    w, h, d = image.shape
    pre_count = np.zeros_like(image, dtype=np.float32)
    predict = np.zeros_like(image, dtype=np.float32)

    x_list = np.squeeze(np.concatenate((np.arange(0, w - patch_size, patch_size[0] // spacing)[:, np.newaxis],
                                        np.array([w - patch_size])[:, np.newaxis])).astype(np.int))
    y_list = np.squeeze(np.concatenate((np.arange(0, h - patch_size, patch_size[1] // spacing)[:, np.newaxis],
                                        np.array([h - patch_size])[:, np.newaxis])).astype(np.int))
    z_list = np.squeeze(np.concatenate((np.arange(0, d - patch_size, patch_size[2] // spacing)[:, np.newaxis],
                                        np.array([d - patch_size])[:, np.newaxis])).astype(np.int))

    start_time = time.time()

    for x in x_list:
        for y in y_list:
            for z in z_list:
                image_patch = image[x:x + patch_size, y:y + patch_size, z:z + patch_size].astype(np.float32)
                image_patch = image_patch[np.newaxis, np.newaxis, ...]

                if args.net[:8] == 'AtlasSeg':
                    atlas_image_path = data_dict['atlas_image_path']
                    atlas_label_path = data_dict['atlas_label_path']
                    atlas_image = nib.load(atlas_image_path).get_fdata()[x:x + patch_size, y:y + patch_size, z:z + patch_size].astype(np.float32)[np.newaxis, np.newaxis, ...]
                    atlas_label = nib.load(atlas_label_path).get_fdata()[x:x + patch_size, y:y + patch_size, z:z + patch_size].astype(np.float32)[np.newaxis, np.newaxis, ...]
                    image_patch = np.concatenate([image_patch, atlas_image, atlas_label], axis=1)
                elif args.net[:2] == 'GA':
                    GA_path = data_dict['GA_path']
                    with open(GA_path, "r") as f:
                        GA = f.read()

                if args.net[:2] == 'GA':
                    patch_tensor = torch.from_numpy(image_patch).cuda()
                    predict_temp = net(patch_tensor)

                    GA_pred = predict_temp[1].squeeze().cpu().data.numpy()
                    predict[x:x + patch_size, y:y + patch_size, z:z + patch_size] += predict_temp[0].squeeze().cpu().data.numpy()
                    pre_count[x:x + patch_size, y:y + patch_size, z:z + patch_size] += 1
                else:
                    patch_tensor = torch.from_numpy(image_patch).cuda()
                    predict[x:x + patch_size, y:y + patch_size, z:z + patch_size] += net(patch_tensor).squeeze().cpu().data.numpy()
                    pre_count[x:x + patch_size, y:y + patch_size, z:z + patch_size] += 1

    predict /= pre_count

    predict = np.squeeze(predict)
    image = np.squeeze(image)

    predict[predict > 0.5] = 1
    predict[predict < 0.5] = 0

    if args.net[:2] == 'GA':
        dice = dc(predict, y_gt)
        mae = float(abs(float(GA_pred) - float(GA)))
        mae_sum += mae
        dice_sum += dice
    else:
        dice = dc(predict, y_gt)
        dice_sum += dice

    image_nii = nib.Nifti1Image(image, affine=None)
    predict_nii = nib.Nifti1Image(predict, affine=None)

    check_dir(os.path.join(save_path, '{}'.format(idx)))
    nib.save(image_nii, os.path.join(save_path, '{}/image.nii.gz'.format(idx)))
    nib.save(predict_nii, os.path.join(save_path, '{}/predict.nii.gz'.format(idx)))

    Dice[idx, 0] = dice
DICE[:, 0] = Dice[:, 0]
excel_path = '/home/xuhaoan/Program/fetal_segmentation/GA-UNet/DouNet/output/excel/' + args.net + args.suffix + '.xlsx'
writer = pd.ExcelWriter(path=excel_path)
data = pd.DataFrame(DICE)
data.to_excel(writer, float_format='%.5f')
writer.save()
writer.close()

if args.net[:2] == 'GA':
    print('Average GA_mae: ', mae_sum / 32.)
print('Average dice: ', dice_sum / 32.)

if args.max is not None:
    for i in range(args.min, args.max + 1):
        if i % 1 == 0:
            net.load_state_dict(torch.load(output_path + '/ckpt/train_' + str(i * 100) + '.pth.gz'))
            net.eval()
            test_meter = AvgMeter()
            dice_sum = 0
            mae_sum = 0
            for idx, data_dict in enumerate(test_list):
                image_path = data_dict['image_path']
                label_path = data_dict['label_path']
                name = image_path.split('/')[-1]
                image = nib.load(image_path).get_fdata()
                y_gt = nib.load(label_path).get_fdata()

                w, h, d = image.shape
                pre_count = np.zeros_like(image, dtype=np.float32)
                predict = np.zeros_like(image, dtype=np.float32)

                x_list = np.squeeze(
                    np.concatenate((np.arange(0, w - patch_size, patch_size[0] // spacing)[:, np.newaxis],
                                    np.array([w - patch_size])[:, np.newaxis])).astype(np.int))
                y_list = np.squeeze(
                    np.concatenate((np.arange(0, h - patch_size, patch_size[1] // spacing)[:, np.newaxis],
                                    np.array([h - patch_size])[:, np.newaxis])).astype(np.int))
                z_list = np.squeeze(
                    np.concatenate((np.arange(0, d - patch_size, patch_size[2] // spacing)[:, np.newaxis],
                                    np.array([d - patch_size])[:, np.newaxis])).astype(np.int))

                start_time = time.time()

                for x in x_list:
                    for y in y_list:
                        for z in z_list:
                            image_patch = image[x:x + patch_size, y:y + patch_size, z:z + patch_size].astype(np.float32)
                            image_patch = image_patch[np.newaxis, np.newaxis, ...]

                            if args.net[:8] == 'AtlasSeg':
                                atlas_image_path = data_dict['atlas_image_path']
                                atlas_label_path = data_dict['atlas_label_path']
                                atlas_image = nib.load(atlas_image_path).get_fdata()[x:x + patch_size, y:y + patch_size,
                                              z:z + patch_size].astype(np.float32)[np.newaxis, np.newaxis, ...]
                                atlas_label = nib.load(atlas_label_path).get_fdata()[x:x + patch_size, y:y + patch_size,
                                              z:z + patch_size].astype(np.float32)[np.newaxis, np.newaxis, ...]
                                image_patch = np.concatenate([image_patch, atlas_image, atlas_label], axis=1)
                            elif args.net[:2] == 'GA':
                                GA_path = data_dict['GA_path']
                                with open(GA_path, "r") as f:
                                    GA = f.read()

                            if args.net[:2] == 'GA':
                                patch_tensor = torch.from_numpy(image_patch).cuda()
                                predict_temp = net(patch_tensor)

                                GA_pred = predict_temp[1].squeeze().cpu().data.numpy()
                                predict[x:x + patch_size, y:y + patch_size, z:z + patch_size] += predict_temp[
                                    0].squeeze().cpu().data.numpy()
                                pre_count[x:x + patch_size, y:y + patch_size, z:z + patch_size] += 1
                            else:
                                patch_tensor = torch.from_numpy(image_patch).cuda()
                                predict[x:x + patch_size, y:y + patch_size, z:z + patch_size] += net(
                                    patch_tensor).squeeze().cpu().data.numpy()
                                pre_count[x:x + patch_size, y:y + patch_size, z:z + patch_size] += 1

                predict /= pre_count

                predict = np.squeeze(predict)
                image = np.squeeze(image)

                predict[predict > 0.5] = 1
                predict[predict < 0.5] = 0

                if args.net[:2] == 'GA':
                    dice = dc(predict, y_gt)
                    mae = float(abs(float(GA_pred) - float(GA)))
                    mae_sum += mae
                    dice_sum += dice
                else:
                    dice = dc(predict, y_gt)
                    dice_sum += dice
                Dice[idx, 0] = dice

            DICE[:, i] = Dice[:, 0]

            excel_path = '/home/xuhaoan/Program/fetal_segmentation/GA-UNet/DouNet/output/excel/' + args.net + args.suffix + '.xlsx'
            writer = pd.ExcelWriter(path=excel_path)
            data = pd.DataFrame(DICE)
            data.to_excel(writer, float_format='%.5f')
            writer.save()
            writer.close()

            if args.net[:2] == 'GA':
                print('Average GA_mae for epoch of ', i * 100, ':', mae_sum / 32.)
            print('Average dice for epoch of ', i * 100, ':', dice_sum / 32.)
