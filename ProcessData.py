import SimpleITK as sitk
import os
import numpy as np
# image_path = 'D:/Data/fetal_segmentation/GA-UNet/resized_datasets/val_data/images3/'
# label_path = 'D:/Data/fetal_segmentation/GA-UNet/resized_datasets/val_data/labels3/'
#
# output_path = 'D:/Data/fetal_segmentation/GA-UNet/DouNet/dataset/val/'
#
# names=os.listdir(image_path)
#
#
# def normlize_min_max(tmp):
#     tmp_max = np.amax(tmp)
#     tmp_min = np.amin(tmp)
#     tmp = (tmp - tmp_min) / (tmp_max - tmp_min)
#     return tmp
#
#
# for name in names:
#     name_temp = name.split('.')[0]
#     if not os.path.isdir(output_path+name_temp):
#         os.mkdir(output_path+name_temp)
#     img=sitk.ReadImage(image_path+name)
#     nda=sitk.GetArrayFromImage(img)
#     nda = normlize_min_max(nda)
#     img_new = sitk.GetImageFromArray(nda)
#     img_new.SetOrigin(img.GetOrigin())
#     img_new.SetDirection(img.GetDirection())
#     img_new.SetSpacing(img.GetSpacing())
#     lbl=sitk.ReadImage(label_path+name_temp+'_tissue_labels.nii.gz')
#     sitk.WriteImage(img_new,output_path+name_temp+'/image.nii.gz')
#     sitk.WriteImage(lbl, output_path + name_temp + '/label.nii.gz')
#

image_path = 'D:/Data/fetal_segmentation/GA-UNet/resized_datasets/val_data/images3/'

label_path = 'D:/Data/fetal_segmentation/GA-UNet/resized_datasets/val_data/labels3/'

output_path = 'D:/Data/fetal_segmentation/GA-UNet/DouNet/dataset/resized/val/'
atlas_path = 'D:/Data/atlas/resized_atlas/'

names=os.listdir(image_path)


def normlize_min_max(tmp):
    tmp_max = np.amax(tmp)
    tmp_min = np.amin(tmp)
    tmp = (tmp - tmp_min) / (tmp_max - tmp_min)
    return tmp


for name in names:
    name_temp = name.split('.')[0]
    if not os.path.isdir(output_path + name_temp):
        os.mkdir(output_path + name_temp)
    # GA_int = int(name[:2]) + (1 if int(name[3]) > 3 else 0)
    # if GA_int == 39:
    #     GA_int = 38
    # atlas_image = sitk.ReadImage(atlas_path + str(GA_int) + '.nii')
    # atlas_label = sitk.ReadImage(atlas_path + str(GA_int) + '_label.nii')
    # atlas_image_nda = sitk.GetArrayFromImage(atlas_image)
    # atlas_image_nda = normlize_min_max(atlas_image_nda)
    # atlas_image_new = sitk.GetImageFromArray(atlas_image_nda)
    # atlas_image_new.SetOrigin(atlas_image.GetOrigin())
    # atlas_image_new.SetDirection(atlas_image.GetDirection())
    # atlas_image_new.SetSpacing(atlas_image.GetSpacing())
    # sitk.WriteImage(atlas_image_new, output_path + name_temp + '/atlas_image.nii.gz')
    # sitk.WriteImage(atlas_label, output_path + name_temp + '/atlas_label.nii.gz')
    GA_temp = int(name[:2]) + int(name[3]) / 7
    with open(output_path + name_temp + '/GA.txt', 'w') as f:
        f.write(str(GA_temp))

import numpy as np
import SimpleITK as sitk
import os
from image_process import crop_pad3D, crop_zeros_joint, load_nifit, normlize_min_max, normlize_mean_std
from id_split import idsplit
import h5py
import random
if __name__ == '__main__':

    # # TODO: resize
    # atlas_path = 'D:/Data/atlas/atlas/'
    # atlas_save_path = 'D:/Data/atlas/resized_atlas/'
    # for i in range(23,39):
    #     atlas = sitk.ReadImage(atlas_path+'Atlas_'+str(i)+'w.nii.gz')
    #     label = sitk.ReadImage(atlas_path+'Atlas_'+str(i)+'w_tissue_labels.nii.gz')
    #     nda = sitk.GetArrayFromImage(atlas)
    #     label_nda = sitk.GetArrayFromImage(label)
    #
    #     label_nda[label_nda!=2]=0
    #
    #     nda, label_nda = crop_zeros_joint(nda, label_nda)
    #     max_coord = np.max(np.shape(nda))
    #
    #     nda = crop_pad3D(nda, [max_coord, max_coord, max_coord])
    #     label_nda = crop_pad3D(label_nda, [max_coord, max_coord, max_coord])
    #
    #     nda = resize(nda, (96, 96, 96), anti_aliasing=False, order=1, preserve_range=True)
    #     label_nda = resize(label_nda, (96, 96, 96), anti_aliasing=False, order=1, preserve_range=True)
    #
    #     nda_new = np.ones([96, 96, 96])
    #     nda_new[label_nda < 0.5] = 0
    #
    #     image = sitk.GetImageFromArray(nda)
    #     image.SetSpacing(atlas.GetSpacing())
    #     image.SetOrigin(atlas.GetOrigin())
    #     image.SetDirection(atlas.GetDirection())
    #     sitk.WriteImage(image, atlas_save_path + str(i) + '.nii')
    #
    #     label_new = sitk.GetImageFromArray(nda_new)
    #     label_new.SetSpacing(label.GetSpacing())
    #     label_new.SetOrigin(label.GetOrigin())
    #     label_new.SetDirection(label.GetDirection())
    #     sitk.WriteImage(label_new, atlas_save_path + str(i) + '_label.nii')
    #


    # TODO: crop
    # TODO: atlas
    # atlas_path = 'D:/Data/atlas/atlas/'
    # atlas_save_path = 'D:/Data/atlas/cropped_atlas/'
    # for i in range(23,39):
    #     atlas = sitk.ReadImage(atlas_path+'Atlas_'+str(i)+'w.nii.gz')
    #     label = sitk.ReadImage(atlas_path+'Atlas_'+str(i)+'w_tissue_labels.nii.gz')
    #     nda = sitk.GetArrayFromImage(atlas)
    #     label_nda = sitk.GetArrayFromImage(label)
    #
    #     label_nda[label_nda!=2]=0
    #     label_nda[label_nda == 2] = 1
    #
    #     nda, label_nda = crop_zeros_joint(nda, label_nda)
    #
    #     nda = crop_pad3D(nda, [128,160,128])
    #     label_nda = crop_pad3D(label_nda, [128,160,128])
    #
    #     image = sitk.GetImageFromArray(nda)
    #     image.SetSpacing(atlas.GetSpacing())
    #     image.SetOrigin(atlas.GetOrigin())
    #     image.SetDirection(atlas.GetDirection())
    #     sitk.WriteImage(image, atlas_save_path + str(i) + '.nii')
    #
    #     label_new = sitk.GetImageFromArray(label_nda)
    #     label_new.SetSpacing(label.GetSpacing())
    #     label_new.SetOrigin(label.GetOrigin())
    #     label_new.SetDirection(label.GetDirection())
    #     sitk.WriteImage(label_new, atlas_save_path + str(i) + '_label.nii')

    # TODO: split
    img_path = 'D:/Data/fetal_segmentation/GA-UNet/raw_datasets/all_data/images/'
    lbl_path = 'D:/Data/fetal_segmentation/GA-UNet/raw_datasets/all_data/labels/'
    CP_path = 'D:/Data/fetal_segmentation/GA-UNet/raw_datasets/all_data/labels_CP/'
    save_path = 'D:/Data/fetal_segmentation/GA-UNet/cropped_datasets/'

    names = os.listdir(img_path)
    random.shuffle(names)
    for i in range(102):
        img = sitk.ReadImage(img_path+names[i])
        img_nda = sitk.GetArrayFromImage(img)
        img_nda = normlize_min_max(img_nda)

        name_temp = names[i].split('.')[0]
        lbl = sitk.ReadImage(lbl_path + name_temp + '_tissue_labels.nii')
        lbl_nda = sitk.GetArrayFromImage(lbl)

        CP = sitk.ReadImage(CP_path + name_temp + '_tissue_labels.nii')
        CP_nda = sitk.GetArrayFromImage(CP)
        temp = img_nda
        img_nda, lbl_nda = crop_zeros_joint(img_nda, lbl_nda)
        _, CP_nda = crop_zeros_joint(temp, CP_nda)
        img_nda = crop_pad3D(img_nda, [128, 160, 128])
        lbl_nda = crop_pad3D(lbl_nda, [128, 160, 128])
        CP_nda = crop_pad3D(CP_nda, [128, 160, 128])

        new_img = sitk.GetImageFromArray(img_nda)
        new_img.SetSpacing(img.GetSpacing())
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetDirection(img.GetDirection())

        new_lbl = sitk.GetImageFromArray(lbl_nda)
        new_lbl.SetSpacing(lbl.GetSpacing())
        new_lbl.SetOrigin(lbl.GetOrigin())
        new_lbl.SetDirection(lbl.GetDirection())

        new_CP = sitk.GetImageFromArray(CP_nda)
        new_CP.SetSpacing(CP.GetSpacing())
        new_CP.SetOrigin(CP.GetOrigin())
        new_CP.SetDirection(CP.GetDirection())


        if i < 60:
            sitk.WriteImage(new_img,save_path+'trainset/images/'+name_temp+'.nii.gz')
            sitk.WriteImage(new_lbl, save_path + 'trainset/labels/' + name_temp + '_labels.nii.gz')
            sitk.WriteImage(new_CP, save_path + 'trainset/labels_CP/' + name_temp + '_CP.nii.gz')

        elif i < 70:
            sitk.WriteImage(new_img, save_path + 'valset/images/' + name_temp + '.nii.gz')
            sitk.WriteImage(new_lbl, save_path + 'valset/labels/' + name_temp + '_labels.nii.gz')
            sitk.WriteImage(new_CP, save_path + 'valset/labels_CP/' + name_temp + '_CP.nii.gz')
        else:
            sitk.WriteImage(new_img, save_path + 'testset/images/' + name_temp + '.nii.gz')
            sitk.WriteImage(new_lbl, save_path + 'testset/labels/' + name_temp + '_labels.nii.gz')
            sitk.WriteImage(new_CP, save_path + 'testset/labels_CP/' + name_temp + '_CP.nii.gz')

    # # TODO: dou_data
    #
    # path = 'D:/Data/fetal_segmentation/GA-UNet/cropped_datasets/'
    # atlas_path = 'D:/Data/atlas/cropped_atlas/'
    # save_path ='D:/Data/fetal_segmentation/GA-UNet/DouNet/dataset/cropped/'
    # sets=['test','train','val']
    # for set in sets:
    #     new_path = path+set+'set/images/'
    #     names = os.listdir(new_path)
    #     for name in names:
    #         name_temp = name.split('.')[0]
    #         img = sitk.ReadImage(new_path+name)
    #         lbl = sitk.ReadImage(path+set+'set/labels/'+name_temp+'_labels.nii.gz')
    #         CP = sitk.ReadImage(path+set+'set/labels_CP/' + name_temp + '_CP.nii.gz')
    #         GA_int = int(name[:2]) + (1 if int(name[3]) > 3 else 0)
    #         if GA_int == 39:
    #             GA_int = 38
    #         atlas_image = sitk.ReadImage(atlas_path + str(GA_int) + '.nii')
    #
    #         # nda = sitk.GetArrayFromImage(img)
    #         # if nda.dtype == np.float64:
    #         #     nda=nda.astype(np.float32)
    #         #     new_img = sitk.GetImageFromArray(nda)
    #         #     new_img.SetSpacing(img.GetSpacing())
    #         #     new_img.SetOrigin(img.GetOrigin())
    #         #     new_img.SetDirection(img.GetDirection())
    #         #
    #         #     atlas_image = sitk.HistogramMatching(atlas_image, new_img)
    #         # else:
    #         #     atlas_image = sitk.HistogramMatching(atlas_image, img)
    #
    #         print(name_temp,' done!')
    #
    #         atlas_label = sitk.ReadImage(atlas_path + str(GA_int) + '_label.nii')
    #
    #         if not os.path.isdir(save_path+set+'/'+name_temp):
    #             os.mkdir(save_path+set+'/'+name_temp)
    #         sitk.WriteImage(img, save_path+set+'/'+name_temp+'/image.nii')
    #         sitk.WriteImage(lbl, save_path + set + '/' + name_temp + '/label_all.nii')
    #         sitk.WriteImage(CP, save_path + set + '/' + name_temp + '/label.nii')
    #         sitk.WriteImage(atlas_image, save_path + set + '/' + name_temp + '/atlas_image.nii.gz')
    #         sitk.WriteImage(atlas_label, save_path + set + '/' + name_temp + '/atlas_label.nii.gz')
    #         GA_temp = int(name[:2]) + int(name[3]) / 7
    #         with open(save_path + set + '/' + name_temp + '/GA.txt', 'w') as f:
    #             f.write(str(GA_temp))
