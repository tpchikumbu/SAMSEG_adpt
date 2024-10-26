import os
import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from torch.nn.functional import threshold, normalize
import src.utils as utils

from src.brats_dataset import BratsDataset, collate_fn
from src.brats_processor import Samprocessor, find_slices, max_slice
from monai.metrics import DiceMetric


from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
"""
This file is used to train a LoRA_sam model. I use that monai DiceLoss for the training. The batch size and number of epochs are taken from the configuration file.
The model is saved at the end as a safetensor.

"""
# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Take dataset path
train_dataset_path = "/home/peter/Documents/Code/samseg/src/testSamples/MEDIUM_Samples"
valid_dataset_path = "/home/peter/Documents/Code/samseg/src/testSamples/MEDIUM_Samples"

# Setup output directories
out_dir = "./train-out"
latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
training_loss_path = os.path.join(out_dir, 'training_loss.csv')
backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')
if not os.path.exists(backup_ckpts_dir):
    os.makedirs(backup_ckpts_dir)
    os.system(f'chmod a+rwx {backup_ckpts_dir}')

# Load SAM model and Create SAM LoRA
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
#Create SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])
model = sam_lora.sam

# Process the dataset
processor = Samprocessor(model)

# Training and validation loaders
train_ds = BratsDataset(train_dataset_path, "train")
valid_ds = BratsDataset(valid_dataset_path, "train")

# Create a dataloader
train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Initialize optimize and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, include_background=False, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

#loss_functions = [nn.MSELoss(), nn.CrossEntropyLoss()]
#loss_weights = [0.4, 0.7]

# Setup output directories
out_dir = "./train-out"
backup_interval = 5
validation_interval = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model to train and into the device
model.train()
model.to(device)

total_loss = []

for epoch in range(num_epochs):
    epoch_losses = []
    dice = DiceMetric(include_background=False, get_not_nans=True, reduction="mean")

    #Training loop
    for i, batch in enumerate(tqdm(train_dataloader)):
      torch.cuda.empty_cache()
      # print(f"{batch[0][0]}:")
      
      slice_idx = max_slice((batch[0][2] > 0).float())
      batch_loss = []
      for idx in slice_idx:
        
        outputs = []
        #with torch.no_grad():
        for image in batch[0][1]:
          input = [processor(image, batch[0][2], idx)]
          chunked_outputs = model(batched_input=input,
                        multimask_output=False)
          #chunked_outputs.requires_grad_(True)
          outputs.extend(chunked_outputs)
          del chunked_outputs
        stk_gt, stk_out = utils.stacking_batch(input, outputs)
        stk_out = stk_out.squeeze(1)
        # stk_gt = stk_gt.unsqueeze(1) # We need to get the [B, C, H, W] starting from [H, W]
        
        # Apply loss function to each scan type and calculate the average
        # dumb_loss = []
        # for i in range(stk_out.shape[0]):
        #   loss_one = seg_loss(stk_out[i], stk_gt.float().to(device))
        #   dumb_loss.append(loss_one)
        # loss_avg = torch.mean(torch.stack(dumb_loss))
        # print(f"Loss (avg): {loss_avg}")

        # Loss calculation with map function
        stk_gt = stk_gt.unsqueeze(1) # We need to get the [B, C, H, W] starting from [H, W]
        stk_gt = stk_gt.repeat(4, 1, 1, 1)
        loss = map(seg_loss, stk_out, stk_gt.float().to(device)) # map(seg_loss, stk_out, stk_gt.float().to(device))
        loss_list = list(loss) # Convert map object to list
        loss_avg = torch.mean(torch.stack(loss_list)) # Calculate the average loss per scan type

        # Optimize parameters with the average loss
        optimizer.zero_grad()
        #loss_avg.backward()
        #optimizer.step()
        

        batch_loss.append(loss_avg)
                
      img_loss = torch.mean(torch.stack(batch_loss))
      # optimizer.zero_grad()
      # scan_loss.requires_grad_(True)
      # scan_loss.backward()
      # # print("GRADIENT:   ", scan_loss.grad)
      # # optimize
      # optimizer.step()
      epoch_losses.append(img_loss.item())

      torch.cuda.empty_cache()

    # Validation loop
    if epoch % validation_interval == 0:
        model.eval()
        valid_loss = []
        valid_dice = []
        predictor = SamPredictor(model)
        for batch in tqdm(valid_dataloader):
          filename, images, mask = batch[0]
          
          output = torch.zeros_like(mask)
          # slice_idx = find_slices((mask > 0).float())
          # print(filename, len(slice_idx))
          # Make the prediction
          torch.cuda.empty_cache()
          for image in images[:1]:
            batch_input = processor(image, mask)
            for scan_slice in (batch_input[:1]):
              predictor.set_torch_image(scan_slice["image"], scan_slice["original_size"])
              pred_mask, iou, low_res = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=scan_slice["boxes"],
                multimask_output=False,
              )
              pred_mask = pred_mask.squeeze(0)
              output[:, :, :, scan_slice["index"]] = torch.max(pred_mask, output[:, :, :, scan_slice["index"]]) 
              # print("Prediction done")
          # Calculate the loss
          loss = seg_loss(output, mask)
          valid_loss.append(loss.item())

          # Calculate dice
          # preds = (torch.sigmoid(low_res) > 0.5).float()
          dice_val = dice(output, mask.float().to(device))
          valid_dice.append(dice_val.item())

          del output, loss, pred_mask, iou, low_res, scan_slice, batch_input, batch, image, filename, images, mask
          print("Memory freed")
        model.train()

        print(f"Validation loss: {mean(valid_loss)}")
        print(f"Validation dice: {sum(valid_dice)}")

  #  print(f'EPOCH: {epoch}; Mean training loss: {mean(epoch_losses)}')
    #utils.save_tloss_csv(training_loss_path, epoch, mean(epoch_losses))
    print("Saving checkpoint...")
    checkpoint = {
      'epoch': epoch,
      'model_sd': model.state_dict(),
      'optim_sd': optimizer.state_dict(),
      'model': model,
    }
    #torch.save(checkpoint, latest_ckpt_path)
    if epoch % backup_interval == 0:
       print(f"Saving backup checkpoint for epoch {epoch}...")
        #torch.save(checkpoint, os.path.join(backup_ckpts_dir, f'epoch{epoch}.pth.tar'))
    print('Checkpoint saved successfully.')

# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]
sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
