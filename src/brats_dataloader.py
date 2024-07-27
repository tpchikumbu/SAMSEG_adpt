import torch
import glob
import os 
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import nibabel as nib

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils
import yaml


class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt 
    
    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()
        if mode == "train":
            self.img_files = glob.glob(os.path.join("/home/peter/Documents/Code/samseg/src/TumourSamples/GOOD_Samples",'*'))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(img_path, os.path.basename(img_path) + "-seg.nii.gz")) 

        else:
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TEST_PATH"],'images','*.jpg'))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(config_file["DATASET"]["TEST_PATH"],'masks', os.path.basename(img_path)[:-4] + ".jpg"))


        self.processor = processor

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> list:
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            # get image and mask in PIL format
            image =  nib.load(img_path)
            mask = nib.load(mask_path)
            # mask = mask.convert('1')
            ground_truth_mask =  np.array(mask)
            original_size = tuple(image.size)[::-1]
    
            # get bounding box prompt
            box = utils.get_bounding_box(ground_truth_mask)
            inputs = self.processor(image, original_size, box)
            inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

            return inputs
    
def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset
    
    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)