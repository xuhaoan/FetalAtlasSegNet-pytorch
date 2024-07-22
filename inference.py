import os
import time

import nibabel as nib
import numpy as np

import torch
from utils import AvgMeter, check_dir
from data_loader import get_list
from medpy.metric.binary import dc, hd95, asd
import argparse

torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--net', default="AtlasSeg", type=str)
parser.add_argument('--best', default=None, type=int)
args = parser.parse_args()

output_path = '/output/' + args.net + '/'
path = '/dataset/'
_, _, test_list = get_list(dir_path=path,net_type=args.net)
net = eval(args.net)().cuda()
if args.best is None:
    net.load_state_dict(torch.load(output_path + '/ckpt/best_val.pth.gz'))
else:
    net.load_state_dict(torch.load(output_path + '/ckpt/train_' + str(args.best) + '.pth.gz'))

patch_size = [96, 96, 96]
spacing = 1

if args.best is None:
    save_path = os.path.join(output_path, 'save_data')
else:
    save_path = os.path.join(output_path, 'save_data' + str(args.best))

check_dir(save_path)

net.eval()

test_meter = AvgMeter()

test_list = sorted(test_list, key=lambda x: x['image_path'])

dice_sum = 0
HD95_sum = 0
ASD_sum = 0

for idx, data_dict in enumerate(test_list):
    image_path = data_dict['image_path']
    label_path = data_dict['label_path']
    name = image_path.split('/')[-1]
    image = nib.load(image_path).get_fdata()
    y_gt = nib.load(label_path).get_fdata()

    w, h, d = image.shape
    pre_count = np.zeros_like(image, dtype=np.float32)
    predict = np.zeros_like(image, dtype=np.float32)

    x_list = np.squeeze(np.concatenate((np.arange(0, w - patch_size[0], patch_size[0] // spacing)[:, np.newaxis],
                                        np.array([w - patch_size[0]])[:, np.newaxis])).astype(np.int))
    y_list = np.squeeze(np.concatenate((np.arange(0, h - patch_size[1], patch_size[1] // spacing)[:, np.newaxis],
                                        np.array([h - patch_size[1]])[:, np.newaxis])).astype(np.int))
    z_list = np.squeeze(np.concatenate((np.arange(0, d - patch_size[2], patch_size[2] // spacing)[:, np.newaxis],
                                        np.array([d - patch_size[2]])[:, np.newaxis])).astype(np.int))

    start_time = time.time()
    for x in x_list:
        for y in y_list:
            for z in z_list:
                image_patch = image[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]].astype(np.float32)
                image_patch = image_patch[np.newaxis, np.newaxis, ...]

                if args.net[:8] == 'AtlasSeg':
                    atlas_image_path = data_dict['atlas_image_path']
                    atlas_label_path = data_dict['atlas_label_path']
                    atlas_image = nib.load(atlas_image_path).get_fdata()[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]].astype(np.float32)[np.newaxis, np.newaxis, ...]
                    atlas_label = nib.load(atlas_label_path).get_fdata()[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]].astype(np.float32)[np.newaxis, np.newaxis, ...]
                    image_patch = np.concatenate([image_patch, atlas_image, atlas_label], axis=1)

                patch_tensor = torch.from_numpy(image_patch).cuda()
                temp = net(patch_tensor)
                predict[x:x + patch_size[0], y:y + patch_size[1],
                z:z + patch_size[2]] += temp.squeeze().cpu().data.numpy()
                pre_count[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] += 1

    predict /= pre_count

    predict = np.squeeze(predict)
    image = np.squeeze(image)

    predict[predict > 0.5] = 1
    predict[predict < 0.5] = 0

    dice = dc(predict, y_gt)
    HD95 = hd95(predict, y_gt)
    ASD = asd(predict, y_gt)

    dice_sum += dice
    HD95_sum += HD95
    ASD_sum += ASD

    image_nii = nib.Nifti1Image(image, affine=None)
    predict_nii = nib.Nifti1Image(predict, affine=None)

    name_temp = data_dict['image_path'].split('/')[-2]
    check_dir(os.path.join(save_path, '{}'.format(name_temp)))
    nib.save(image_nii, os.path.join(save_path, '{}/image.nii.gz'.format(name_temp)))
    nib.save(predict_nii, os.path.join(save_path, '{}/predict.nii.gz'.format(name_temp)))


print('Average dice: ', dice_sum / len(test_list))
print('Average HD95: ', HD95_sum / len(test_list))
print('Average ASD: ', ASD_sum / len(test_list))
