import SimpleITK as sitk
import os
import numpy as np

path = 'D:/Data/atlas/atlas/'
atlas_path = 'D:/Data/atlas/resized_atlas/'

for i in range(23,39):
    img = sitk.ReadImage(path+'Atlas_'+str(i)+'w.nii.gz')
    lbl = sitk.ReadImage(path+'Atlas_'+str(i)+'w_tissue_labels.nii.gz')