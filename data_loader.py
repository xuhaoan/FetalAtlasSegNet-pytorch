import os
import glob

import nibabel as nib

from augmentation import *


def get_list(dir_path, net_type):
    """
    This function is to read dataset from dataset dir.
    The dataset dir should be set as follow:
    -- Data
        -- case1
            -- image.nii.gz
            -- label.nii.gz
        -- case2
        ...
    """
    print("Reading Data...")
    train_dict_list = []
    test_dict_list = []
    val_dict_list = []
    train_path_list = glob.glob(os.path.join(dir_path + 'train/', '*'))
    test_path_list = glob.glob(os.path.join(dir_path + 'test/', '*'))
    val_path_list = glob.glob(os.path.join(dir_path + 'val/', '*'))

    if net_type[:8] == 'AtlasSeg':
        image_name = 'image.nii.gz'
        label_name = 'label.nii.gz'
        atlas_image_name = 'atlas_image.nii.gz'
        atlas_label_name = 'atlas_label.nii.gz'
        for path in train_path_list:
            train_dict_list.append(
                {
                    'image_path': os.path.join(path, image_name),
                    'label_path': os.path.join(path, label_name),
                    'atlas_image_path': os.path.join(path, atlas_image_name),
                    'atlas_label_path': os.path.join(path, atlas_label_name),
                }
            )
        for path in test_path_list:
            test_dict_list.append(
                {
                    'image_path': os.path.join(path, image_name),
                    'label_path': os.path.join(path, label_name),
                    'atlas_image_path': os.path.join(path, atlas_image_name),
                    'atlas_label_path': os.path.join(path, atlas_label_name),
                }
            )
        for path in val_path_list:
            val_dict_list.append(
                {
                    'image_path': os.path.join(path, image_name),
                    'label_path': os.path.join(path, label_name),
                    'atlas_image_path': os.path.join(path, atlas_image_name),
                    'atlas_label_path': os.path.join(path, atlas_label_name),
                }
            )
    else:
        image_name = 'image.nii.gz'
        label_name = 'label.nii.gz'

        for path in train_path_list:
            train_dict_list.append(
                {
                    'image_path': os.path.join(path, image_name),
                    'label_path': os.path.join(path, label_name),
                }
            )
        for path in test_path_list:
            test_dict_list.append(
                {
                    'image_path': os.path.join(path, image_name),
                    'label_path': os.path.join(path, label_name),
                }
            )
        for path in val_path_list:
            val_dict_list.append(
                {
                    'image_path': os.path.join(path, image_name),
                    'label_path': os.path.join(path, label_name),
                }
            )

    train_list = train_dict_list
    val_list = val_dict_list
    test_list = test_dict_list
    print("Finished! Train:{} Val:{} Test:{}".format(len(train_list), len(val_list), len(test_list)))

    return train_list, val_list, test_list


class TrainGenerator(object):
    """
    This is the class to generate the patches
    """

    def __init__(self, data_list, batch_size, patch_size, net_type):
        self.data_list = data_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.net_type = net_type

    def get_item(self):

        dict_list = random.sample(self.data_list, self.batch_size)

        image_list = [dict_item['image_path'] for dict_item in dict_list]
        label_list = [dict_item['label_path'] for dict_item in dict_list]
        if self.net_type[:8] == 'AtlasSeg':
            atlas_image_list = [dict_item['atlas_image_path'] for dict_item in dict_list]
            atlas_label_list = [dict_item['atlas_label_path'] for dict_item in dict_list]
            image_patch, label_patch, _ = self._sample_patch(image_list, label_list, atlas_image_list, atlas_label_list)
        else:
            image_patch, label_patch, _ = self._sample_patch(image_list, label_list)
        return image_patch, label_patch

    def _sample_patch(self, image_list, clean_list, atlas_image_list=None, atlas_label_list=None, GA_list=None):
        w_half_size = self.patch_size[0] // 2
        h_half_size = self.patch_size[1] // 2
        d_half_size = self.patch_size[2] // 2
        image_patch_list = []
        label_patch_list = []
        if self.net_type[:8] == 'AtlasSeg':

            for image_path, clean_path, atlas_image_path, atlas_label_path in zip(image_list, clean_list,
                                                                                  atlas_image_list, atlas_label_list):
                image = nib.load(image_path).get_fdata()
                label = nib.load(clean_path).get_fdata()
                atlas_image = nib.load(atlas_image_path).get_fdata()
                atlas_label = nib.load(atlas_label_path).get_fdata()

                image, label, atlas_image, atlas_label = RandomFlip()(image, label, atlas_image, atlas_label)
                image, label, atlas_image, atlas_label = RandomContrast(alpha=(0.9, 1.1), execution_probability=0.1)(
                    image, label, atlas_image, atlas_label)
                image, label, atlas_image, atlas_label = ElasticDeformation(alpha=(-1000, 1000),
                                                                            execution_probability=0.1)(image, label,
                                                                                                       atlas_image,
                                                                                                       atlas_label)
                image, label, atlas_image, atlas_label = rot_3d(image, label, atlas_image, atlas_label, max_angle=30)

                w, h, d = image.shape

                label_index = np.where(label == 1)
                length_label = label_index[0].shape[0]

                p = random.random()
                if p < 0.875:
                    sample_id = random.randint(1, length_label - 1)
                    x, y, z = label_index[0][sample_id], label_index[1][sample_id], label_index[2][sample_id]
                else:
                    x, y, z = random.randint(0, w), random.randint(0, h), random.randint(0, d)

                if x < w_half_size:
                    x = w_half_size
                elif x > w - w_half_size:
                    x = w - w_half_size - 1

                if y < h_half_size:
                    y = h_half_size
                elif y > h - h_half_size:
                    y = h - h_half_size - 1

                if z < d_half_size:
                    z = d_half_size
                elif z > d - d_half_size:
                    z = d - d_half_size - 1

                image_patch = image[x - w_half_size:x + w_half_size, y - h_half_size:y + h_half_size,
                              z - d_half_size:z + d_half_size].astype(np.float32)[np.newaxis, np.newaxis, ...]
                label_patch = label[x - w_half_size:x + w_half_size, y - h_half_size:y + h_half_size,
                              z - d_half_size:z + d_half_size].astype(np.float32)
                atlas_image_patch = atlas_image[x - w_half_size:x + w_half_size, y - h_half_size:y + h_half_size,
                              z - d_half_size:z + d_half_size].astype(np.float32)[np.newaxis, np.newaxis, ...]
                atlas_label_patch = atlas_label[x - w_half_size:x + w_half_size, y - h_half_size:y + h_half_size,
                              z - d_half_size:z + d_half_size].astype(np.float32)[np.newaxis, np.newaxis, ...]

                image_output = np.concatenate((image_patch, atlas_image_patch, atlas_label_patch), axis=1)
                image_patch_list.append(image_output)
                label_patch_list.append(label_patch[np.newaxis, np.newaxis, ...])

            image_out = np.concatenate(image_patch_list, axis=0)
            label_out = np.concatenate(label_patch_list, axis=0)

            return image_out, label_out, None

        else:
            for image_path, clean_path in zip(image_list, clean_list):
                image = nib.load(image_path).get_fdata()
                label = nib.load(clean_path).get_fdata()

                image, label, _, _ = RandomFlip()(image, label)
                image, label, _, _ = RandomContrast(alpha=(0.9, 1.1), execution_probability=0.1)(image, label)
                image, label, _, _ = ElasticDeformation(alpha=(-1000, 1000), execution_probability=0.1)(image, label)
                image, label, _, _ = rot_3d(image, label, max_angle=30)

                w, h, d = image.shape
                label_index = np.where(label == 1)
                length_label = label_index[0].shape[0]
                p = random.random()
                if p < 0.875:
                    sample_id = random.randint(1, length_label - 1)
                    x, y, z = label_index[0][sample_id], label_index[1][sample_id], label_index[2][sample_id]
                else:
                    x, y, z = random.randint(0, w), random.randint(0, h), random.randint(0, d)

                if x < w_half_size:
                    x = w_half_size
                elif x > w - w_half_size:
                    x = w - w_half_size - 1

                if y < h_half_size:
                    y = h_half_size
                elif y > h - h_half_size:
                    y = h - h_half_size - 1

                if z < d_half_size:
                    z = d_half_size
                elif z > d - d_half_size:
                    z = d - d_half_size - 1

                image_patch = image[x - w_half_size:x + w_half_size, y - h_half_size:y + h_half_size,
                              z - d_half_size:z + d_half_size].astype(np.float32)
                label_patch = label[x - w_half_size:x + w_half_size, y - h_half_size:y + h_half_size,
                              z - d_half_size:z + d_half_size].astype(np.float32)

                image_patch_list.append(image_patch[np.newaxis, np.newaxis, ...])
                label_patch_list.append(label_patch[np.newaxis, np.newaxis, ...])

            image_out = np.concatenate(image_patch_list, axis=0)
            label_out = np.concatenate(label_patch_list, axis=0)

            return image_out, label_out, None

