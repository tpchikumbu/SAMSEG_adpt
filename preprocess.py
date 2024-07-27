# Source: https://github.com/KurtLabUW/brats2023_updated/blob/master/processing/preprocess.py
# Date: July 4, 2024

import numpy as np
from skimage import exposure

def znorm_rescale(img):
    """Applies Z-score normalization and rescaling to a MRI image."""

    #-----------------------------------------------------------------------------
    # Step 1: Z-Score Normalisation
    # Create a copy of the input image to avoid modifying the original.
    movingNan = np.copy(img)

    # Replace zeros with NaN to avoid them affecting the mean and std calculation.
    movingNan[movingNan == 0] = np.nan

    # Compute the mean and standard deviation of the non-zero elements.
    movingMean = np.nanmean(movingNan)
    movingSTD = np.nanstd(movingNan)

    # Apply the actual Z-Score normalisation.
    # Subtract the mean and divide by the standard deviation.
    moving = (img - movingMean) / movingSTD
    #-----------------------------------------------------------------------------
    # Step 2: Linear Transformation
    # Scaling the normalised values to a specific range.
    b = 255 / (1 - (moving.max() / moving.min()))
    a = -b / moving.min()

    # Apply the linear transformation to the normalised values.
    movingNorm = np.copy(moving)
    movingNorm = np.round((movingNorm * a) +b, 2)
    #-----------------------------------------------------------------------------
    # Step 3: Rescaling
    # Rescaling the intensity values to a specified range
    # Compute the 1st and 99th percentiles of the transformed values
    # 1st and 99th may not be optimal, further testing could be done here.
    p2, p98 = np.percentile(movingNorm, (1, 99)) 

    # Rescale the intensity values to the range [0, 1] using these percentiles
    moving_rescale = exposure.rescale_intensity(movingNorm, in_range=(p2, p98))
    #-----------------------------------------------------------------------------

    return moving_rescale

# Crop ranges for center crop.
# X_START, X_END, Y_START, Y_END, Z_START, Z_END = (56,184, 24,216, 14,142)
X_START, X_END, Y_START, Y_END, Z_START, Z_END = (51,189, 19,221, 9,147)

def center_crop(img):
    """Center crops a MRI image (or seg) to be (128, 192, 128)."""
    return img[X_START:X_END, Y_START:Y_END, Z_START:Z_END]

def undo_center_crop(input):
    """Undos center crop of a MRI image (or seg)."""
    out = np.zeros((240, 240, 155))
    out[X_START:X_END, Y_START:Y_END, Z_START:Z_END] = input 
    return out