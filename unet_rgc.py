
import numpy as np
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import torch
import random
import pandas as pd
import os

from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


file_names = Path("/home/akaur101/data/cross_validation/file_names_per_fold")
image_dir = Path(sys.argv[1])
mask_dir = Path(sys.argv[2])
fold = int(sys.argv[3])

unet_seg_dir = Path("/home/akaur101/data/f25/unet/unet_segmentations")

train_image_names = {}
test_image_names = {}

# Iterate over each text file in the folder
for file in file_names.glob('*.txt'):
    fold_num = int(file.name.split('_')[-1].split('.')[0])
    
    with open(file, 'r') as f:
        lines = f.readlines()
        filenames = [filename.strip() + 'png' for filename in lines[0].split('png') if filename.strip()]
        formatted_filenames = [f"{filename}" for filename in filenames]

        if 'train' in file.name:
            train_image_names[fold_num] = formatted_filenames
        elif 'test' in file.name:
            test_image_names[fold_num] = formatted_filenames

test_image_paths = {}
for fold_num, filenames in test_image_names.items():
    test_image_paths[fold_num] = [image_dir / f for f in filenames]

test_mask_paths = {}
for fold_num, filenames in test_image_names.items():
    test_mask_paths[fold_num] = [mask_dir / f for f in filenames]

test_images = test_image_paths[fold]
test_masks = test_mask_paths[fold]

test_images_padded = []
for image in test_images:
    test_image = plt.imread(image)
    test_im_nor = test_image/np.max(test_image)
    test_image_padded = np.pad(test_im_nor, 40, mode='constant')
    test_images_padded.append(test_image_padded)

test_masks_padded = []
gt_masks = []  
for mask in test_masks:
    test_mask = plt.imread(mask)
    test_mas_nor = test_mask/np.max(test_mask)
    test_mask_padded = np.pad(test_mas_nor, 40, mode='constant')
    test_masks_padded.append(test_mask_padded)
    gt_masks.append(test_mas_nor)

# CNN Model classes (your existing code)
class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.conv(x)

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv1 = double_conv(1, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.conv5 = double_conv(512, 1024)
        self.conv6 = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, image):
        x1 = self.conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.conv4(x6)
        x8 = self.maxpool2(x7)
        x9 = self.conv5(x8)
        x10 = self.conv6(x9)
        return x10

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net()
model = model.float()
model = model.to(device=device)
model.load_state_dict(torch.load(f"/home/akaur101/data/f25/2d_rgc/trained_dsa_models/dsa_rgc_bestmodel_fold{fold}_dict.pth", map_location=device))
model.eval()

def img_process(next_cents, padded_image, padded_mask):
    image_list = []
    mask_list = []
    val_list = []
    rows = []
    cols = []

    for i in next_cents:
        row = i[0]
        col = i[1]
        val = padded_image[row][col]
        snap_image = padded_image[row-40:row+40, col-40:col+40]
        snap_mask = padded_mask[row-1:row+2, col-1:col+2]
        image_list.append(snap_image)
        mask_list.append(snap_mask)
        val_list.append(val)
        rows.append(row)
        cols.append(col)
    
    return image_list, mask_list, val_list, rows, cols

class newcelldata(Dataset):
    def __init__(self, image_list, mask_list, rows, cols, transforms):
        self.image_list = image_list
        self.mask_list = mask_list
        self.rows = rows
        self.cols = cols
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index].astype(float)
        mask = self.mask_list[index].astype(float)
        rows = self.rows[index]
        cols = self.cols[index]
        
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        return (image, mask, rows, cols)


transforms_tensor = transforms.ToTensor()
save_dir = Path("/home/akaur101/data/f25/unet_rgc/unet_rgc_results/fold_" + str(fold))
save_dir.mkdir(parents=True, exist_ok=True)


all_results = []

print(f"Testing {len(test_images_padded)} images from fold {fold}")

for img_idx, (test_image_padded, test_mask_padded, original_mask) in enumerate(zip(test_images_padded, test_masks_padded, gt_masks)):
    print(f"\nProcessing image {img_idx + 1}/{len(test_images_padded)}")
    
    # Get the image name
    image_name = test_images[img_idx].stem
    
    # Load UNet segmentation
    unet_seg_path = unet_seg_dir / f"fold_{fold}" / f"{image_name}_unet_seg.npy"
    unet_seg = np.load(unet_seg_path)
    
    # Get foreground pixel coordinates (where segmentation > 0)
    foreground_coords = np.argwhere(unet_seg > 0)
    
    # Create centroid dataframe from foreground pixels
    centroid_df = pd.DataFrame(foreground_coords, columns=['row', 'col'])
    # Add padding offset since test_image_padded has 40-pixel padding
    centroid_df['row'] = centroid_df['row'] + 40
    centroid_df['col'] = centroid_df['col'] + 40
    
    print(f"Loaded {len(centroid_df)} foreground pixels from UNet segmentation")

    
    S = []  # master centroid list
    runn_cent = []
    next_S = []
    df = centroid_df
    all_pred = []
    all_indices = []

    saved_masks = []
    saved_iterations = []


    for i in df.index:
        row = df['row'][i]
        col = df['col'][i]
        next_S.append([row, col])

    S.append(next_S)
    counter = 0
    max_iterations = 100  


    # Iterative detection
    while len(next_S) > 0:
        counter += 1
        print(f"Iteration {counter}, processing {len(next_S)} centroids")
        
        test_image_list, test_mask_list, val_list, rows, cols = img_process(
            next_S, test_image_padded, test_mask_padded)
        
        test_dataset = newcelldata(
            image_list=test_image_list,
            mask_list=test_mask_list,
            rows=rows,
            cols=cols,
            transforms=transforms_tensor)
        
        test_loader = DataLoader(test_dataset, 50, shuffle=False)
       
        with torch.no_grad():
            for k, data in enumerate(test_loader):
                images, masks, rows, cols = data
                images = images.to(device=device)
                masks = masks.to(device=device)
                pred = model(images.float())
                final_pred = torch.sigmoid(pred)
                
                pred_temp = torch.where(torch.squeeze(final_pred) < 0.1, 0, 1)
                if len(final_pred) == 1:
                    pred_temp = torch.where((final_pred) < 0.1, 0, 1)
                
                pred_ind = torch.argwhere(pred_temp)
                
                temp_ind_list = []
                temp_val_list = []
                
                for i in range(len(pred_ind)):
                    current = pred_ind[i][0].item()
                    
                    row_item = rows[current].item()
                    pred_ind_row = pred_ind[i][1].item()
                    x = row_item - 1 + pred_ind_row
        
                    col_item = cols[current].item()
                    pred_ind_col = pred_ind[i][2].item()
                    y = col_item - 1 + pred_ind_col
                    
                    boundary_size = 40
        
                    if boundary_size <= x < test_image_padded.shape[0] - boundary_size and \
                       boundary_size <= y < test_image_padded.shape[1] - boundary_size:
                        val = torch.squeeze(final_pred[current])[pred_ind_row, pred_ind_col]
                        temp_ind_list.append([x, y])
                        temp_val_list.append(val)
                    else:
                        continue
                        
                all_indices.append(temp_ind_list)
                all_pred.append(temp_val_list)

                
            
                for i in temp_ind_list:
                    if i not in runn_cent:
                        runn_cent.append(i)
                
        
        temp_snapshot_mask = np.zeros_like(test_mask_padded)
        for idx in runn_cent:
            temp_snapshot_mask[idx[0], idx[1]] = 1


        saved_masks.append(temp_snapshot_mask[40:-40, 40:-40].copy())
        saved_iterations.append(counter)

        
        next_S = []
        for i in runn_cent:
            if i not in S:
                next_S.append(i)
        S.extend(next_S)
    
    print(f"Completed after {counter} iterations")

    
    recon_mask = np.zeros_like(test_mask_padded)
    flat_list_ind = [item for sublist in all_indices for item in sublist]
    flat_list_pred = [item for sublist in all_pred for item in sublist]
    

    for i in range(len(flat_list_ind)):
        x = flat_list_ind[i][0]
        y = flat_list_ind[i][1]
        recon_mask[x, y] = 1
    
  
    new_recon_mask = recon_mask[40:-40, 40:-40]
    

    np.save(
        save_dir / f"{image_name}_indices.npy",
        np.array(flat_list_ind)
    )

    pred_array = np.array([p.cpu().numpy() if torch.is_tensor(p) else p for p in flat_list_pred])
    np.save(
        save_dir / f"{image_name}_predictions.npy",
        pred_array
    )

    

    np.save(
        save_dir / f"{image_name}_all_snapshots.npy",
        np.array(saved_masks)
    )
    np.save(
        save_dir / f"{image_name}_snapshot_iterations.npy",
        np.array(saved_iterations)
    )
    print(f"Saved {len(saved_masks)} iteration snapshots")

    plt.imshow(new_recon_mask, cmap="gray")
    plt.savefig(save_dir / f"{image_name}_results.png", bbox_inches='tight')
    
    
    
    print(f"Image {image_name}: Detected {len(flat_list_ind)} cells in {counter} iterations")

print(f"\nCompleted testing all {len(test_images_padded)} images")
print(f"Results saved to: {save_dir}")








