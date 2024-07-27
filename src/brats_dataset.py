# Source: https://github.com/KurtLabUW/brats2023_updated/blob/master/datasets/brats_dataset.py
# Date: July 4, 2024

from torch.utils.data import Dataset
import os
import nibabel as nib

from preprocess import znorm_rescale, center_crop

import numpy as np
import torch
import synapseclient

class BratsDataset(Dataset):
    """Dataset class for loading BraTS training and test data.
    
    Args:
        data_dir: Directory of training or test data.
        mode: Either 'train' or 'test' specifying which data is being loaded.
    """
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.subject_list = os.listdir(data_dir)
        self.mode = mode

    def __len__(self):
        return len(self.subject_list)
    
    def load_nifti(self, subject_name, suffix):        
        """Loads nifti file for given subject and suffix."""

        nifti_filename = f'{subject_name}-{suffix}.nii.gz'
        nifti_path = os.path.join(self.data_dir, subject_name, nifti_filename)
        nifti = nib.load(nifti_path)
        return nifti
    
    def load_subject_data(self, subject_name):
        """Loads images (and segmentation if in train mode) and extra info for a subject."""

        modalities_data = []
        for suffix in ['t1c', 't1n', 't2f', 't2w']:
            modality_nifti = self.load_nifti(subject_name, suffix)
            modality_data = modality_nifti.get_fdata()
            modalities_data.append(modality_data)

        if self.mode == 'train':
            seg_nifti = self.load_nifti(subject_name, 'seg')
            seg_data = seg_nifti.get_fdata()
            return modalities_data, seg_data
        elif self.mode == 'test':
            return modalities_data
    
    def __getitem__(self, idx):
        subject_name = self.subject_list[idx]

        # Load the data and extra info.
        if self.mode == 'train':
            imgs, seg = self.load_subject_data(subject_name)
        elif self.mode == 'test':
            imgs = self.load_subject_data(subject_name)

        # Do Z-score norm and rescaling preprocessing.
        imgs = [znorm_rescale(img) for img in imgs]

        # Perform center crop.
        imgs = [center_crop(img) for img in imgs]

        imgs = [x[None, ...] for x in imgs]
        imgs = [np.ascontiguousarray(x, dtype=np.float32) for x in imgs]

        # Convert to torch tensors.
        imgs = [torch.from_numpy(x) for x in imgs]

        # If train mode, process segmentation similarly.
        if self.mode == 'train':
            seg = center_crop(seg)
            seg = seg[None, ...]
            seg = np.ascontiguousarray(seg)
            seg = torch.from_numpy(seg)

            return subject_name, imgs, seg
            # subject_name (str): root of files
            # img (array(tensors)) : collection of images as tensors
            # seg (tensor) : tensor with ground truth mask
        
        elif self.mode == 'test':
            return subject_name, imgs
        
def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset
    
    Return:
        (list): list of batched dataset so a list(dict)
    """
    return batch

def download() :
    "Download dataset from Synapse server"    
    syn = synapseclient.Synapse() 
    syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTcxNTU4NTY1NCwiaWF0IjoxNzE1NTg1NjU0LCJqdGkiOiI3OTYyIiwic3ViIjoiMzUwMTA0MyJ9.LuuYJPJBWtICNUEBDikko1eF2C9QzIIlb9QE8murf9PE0c5ztmVe-WAXD407fvObflffDMBQHfRlmg-8CuL7JrOKFQbqMp9I172s9wULqHACFOyJiNbpfjwgxQ5BSTpR9pVFTxY-XupHgAy4FlGoh_XVDcO3A-tW9plKZO0K3oWimmHUYjQ3awf7kPuLovXvFY5zfP6YyM2vZaOnFJ2MPWUI05Rtlh1WTeWdvDWPYSTlwk6IPmVSrShpLwQXZ35VVBWoiHkj9OD09h91_zlvS0N_sxDIB7-s2qnJiU9oFo3vowLjRZpKM0iUFxN0GENanKC0c4AK8d_Z77gPYEkR_w") 
    
    # Obtain a pointer and download the data 
    syn51514055 = syn.get(entity='syn51514055') 
    
    # Get the path to the local copy of the data file 
    filepath = syn51514055.path
    print(f"Saved data as \'{filepath}\'") 