from src.segment_anything.utils.transforms import ResizeLongestSide
from src.lora import LoRA_sam
import src.utils as utils
import numpy as np
import torch
import PIL
from typing import Optional, Tuple


class Samprocessor:
    """
    Processor that transform the image and bounding box prompt with ResizeLongestSide and then pre process both data
        Arguments:
            sam_model: Model of SAM with LoRA weights initialised
        
        Return:
            inputs (list(dict)): list of dict in the input format of SAM containing (prompt key is a personal addition)
                image: Image preprocessed
                boxes: bounding box preprocessed
                prompt: bounding box of the original image

    """
    def __init__(self, sam_model: LoRA_sam):
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def __call__(self, image, mask) -> list:
        #imgs = [img for img in image]
        seg = (mask > 0).float() # Convert to binary
        slice_idx = find_slices(seg)
        
        seg_slice = [seg[ :, :, :, i] for i in slice_idx]
        prompt = [utils.generate_bbox(tum[0, :, :].cpu().numpy(), margin=0) for tum in seg_slice]
        
        img_slice = [image[ :, :, :, i] for i in slice_idx] #image[ :, :, :, slice_idx]
        # = torch.Size([ 1, 138, 202])
        
        original_size = tuple(seg_slice[0].shape[-2:])
        
        # Processing of the image
        # seg_mask = np.array(seg_slice)#self.process_image(seg_slice, original_size, slice_idx)
        image_torch = [self.process_image(img) for img in img_slice]

        # Transform input prompts
        box_torch = [self.process_prompt(pmt, original_size) for pmt in prompt]

        volume = []
        for j in range(len(slice_idx)):
            inputs = {"image": image_torch[j], # tensor
                    "original_size": original_size, # tuple
                    "boxes": box_torch[j], # tensor
                    "prompt" : prompt[j], #list
                    "ground_truth_mask" : seg_slice[j][0, :, :]} #tensor
            volume.append(inputs)
        return volume


    def process_image(self, image: torch.tensor) -> torch.tensor:
        """
        Preprocess the image to make it to the input format of SAM

        Arguments:
            image: Image loaded in PIL
            original_size: tuple of the original image size (H,W)

        Return:
            (tensor): Tensor of the image preprocessed
        """

        # img_slice = image[:, :, :, :, slice_idx]
        
        # = torch.Size([1, 1, 138, 202])

        # Convert every img to RGB by duplicates the single channel three times.
        img_RGB = torch.cat([image, image, image], dim=0) # img is B3HW
        # Change order of dimensions
        nd_image = img_RGB.permute(1, 2, 0).cpu().numpy()
        # Ensure img_slice is in uint8 format
        if nd_image.dtype != np.uint8:
            nd_image = (nd_image * 255).astype(np.uint8)

        input_image = self.transform.apply_image(nd_image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        return input_image_torch

    def process_prompt(self, box: list, original_size: tuple) -> torch.tensor:
        """
        Preprocess the prompt (bounding box) to make it to the input format of SAM

        Arguments:
            box: Bounding bounding box coordinates in [XYXY]
            original_size: tuple of the original image size (H,W)

        Return:
            (tensor): Tensor of the prompt preprocessed
        """
        # We only use boxes
        box_torch = None
        nd_box = np.array(box).reshape((1,4))
        box = self.transform.apply_boxes(nd_box, original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
        box_torch = box_torch[None, :]

        return box_torch


    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

def max_slice(seg):
# """Finds the slice (in the z-direction) of a segmentation with the most tumour voxels."""
# one_hot_encoded = np.where(seg != 0, 1, 0)
# slice_sums = np.sum(one_hot_encoded, axis=(0,1))
# max_index = np.argmax(slice_sums)

    """Finds the slice with the largest cross-sectional area of the tumor.

    Args:
        seg: The 3D segmentation tensor with shape [B, 1, H, W, D].

    Returns:
        The index of the slice with the largest cross-sectional area of the tumor.
    """
    
    # Ensure seg is a numpy array.
    if not isinstance(seg, np.ndarray):
        seg_np = seg.cpu().numpy()
        seg_np = seg_np.squeeze(0)  # Remove batch and channel dimensions if present.
    else:
        seg_np = seg

    # Ensure seg_np has shape (H, W, D).
    assert len(seg_np.shape) == 3, f"Expected shape (H, W, D), but got {seg_np.shape}"

    # Compute the area of the tumor in each slice along the depth dimension.
    areas = [np.sum(seg_np[:, :, i]) for i in range(seg_np.shape[2])]

    # Find the slice with the largest area.
    largest_slice_idx = np.argmax(areas)

    return [largest_slice_idx]

def find_slices(seg, skip = 0):
# """Finds the slice (in the z-direction) of a segmentation with the most tumour voxels."""
# one_hot_encoded = np.where(seg != 0, 1, 0)
# slice_sums = np.sum(one_hot_encoded, axis=(0,1))
# max_index = np.argmax(slice_sums)

    """Finds the slice with the largest cross-sectional area of the tumor.

    Args:
        seg: The 3D segmentation tensor with shape [B, 1, H, W, D].

    Returns:
        The index of the slice with the largest cross-sectional area of the tumor.
    """
    
    # Ensure seg is a numpy array.
    if not isinstance(seg, np.ndarray):
        seg_np = seg.cpu().numpy()
        seg_np = seg_np.squeeze(0)  # Remove batch and channel dimensions if present.
    else:
        seg_np = seg

    # Ensure seg_np has shape (H, W, D).
    assert len(seg_np.shape) == 3, f"Expected shape (H, W, D), but got {seg_np.shape}"

    # Compute the area of the tumor in each slice along the depth dimension.
    areas = [np.sum(seg_np[:, :, i]) for i in range(0, seg_np.shape[2], 1 + skip)]

    # Find the slice with the largest area.
    slice_idx = np.nonzero(areas)

    return slice_idx[0]