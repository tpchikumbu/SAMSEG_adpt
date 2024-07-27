import os
import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
import torch.nn as nn
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import src.utils as utils

#from src.brats_dataloader import DatasetSegmentation, collate_fn
from src.brats_dataset import BratsDataset, collate_fn
# from src.processor import Samprocessor
from src.brats_processor import Samprocessor, max_slice


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
# Load SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
#Create SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])
model = sam_lora.sam
# Process the dataset
processor = Samprocessor(model)

# Ring loader vs BraTS loader
# train_ds = DatasetSegmentation(config_file, processor, mode="train")
train_ds = BratsDataset(train_dataset_path, "train")

# Create a dataloader
train_dataloader = DataLoader(train_ds, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
# Initialize optimize and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=6e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

loss_functions = [nn.MSELoss(), nn.CrossEntropyLoss()]
loss_weights = [0.4, 0.7]

# Setup output directories
out_dir = "./train-out"
backup_interval = 5

latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
training_loss_path = os.path.join(out_dir, 'training_loss.csv')
backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')
if not os.path.exists(backup_ckpts_dir):
    os.makedirs(backup_ckpts_dir)
    os.system(f'chmod a+rwx {backup_ckpts_dir}')


device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model to train and into the device
model.train()
model.to(device)

total_loss = []

for epoch in range(num_epochs):
    epoch_losses = []

    for i, batch in enumerate(tqdm(train_dataloader)):
      torch.cuda.empty_cache()
      print(f"{batch[0][0]}:")
      batch_loss = []
      input = []
      for image in batch[0][1]:
        input += (processor(image, batch[0][2]))
      outputs = model(batched_input=input,
                      multimask_output=False)

      stk_gt, stk_out = utils.stacking_batch(input, outputs)
      stk_out = stk_out.squeeze(1)
      stk_gt = stk_gt.unsqueeze(1) # We need to get the [B, C, H, W] starting from [H, W]
      loss = seg_loss(stk_out, stk_gt.float().to(device))
      #loss = utils.compute_loss(stk_out, stk_gt, loss_functions, loss_weights, device)
      print(f"Loss: {loss}")        
      
      optimizer.zero_grad()
      loss.backward()
      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}; Mean training loss: {mean(epoch_losses)}')
    utils.save_tloss_csv(training_loss_path, epoch, mean(epoch_losses))
    print("Saving checkpoint...")
    checkpoint = {
      'epoch': epoch,
      'model_sd': model.state_dict(),
      'optim_sd': optimizer.state_dict(),
      'model': model,
      'loss_functions': loss_functions,
      'loss_weights': loss_weights,
    }
    torch.save(checkpoint, latest_ckpt_path)
    if epoch % backup_interval == 0:
        torch.save(checkpoint, os.path.join(backup_ckpts_dir, f'epoch{epoch}.pth.tar'))
    print('Checkpoint saved successfully.')

# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]
sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
