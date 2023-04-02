from metrics import fid, inception
from tqdm import tqdm
import os
import pandas as pd
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from BDInvert.image_tools import *
import numpy as np
import scipy

from metrics.arcface import Backbone
from metrics.kid import compute_kid
from metrics.fid import compute_fid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class BaseDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 key,
                 resolution=1024,
                 transform=None,
                 target_transform=None,
                 data_format='csv',
                 **_unused_kwargs):
        self.meta_csv = pd.read_csv(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.resolution = resolution
        self.key = key
        self.device = device

    def __len__(self):
        return len(self.meta_csv)

    def load_image(self,img_path):
        image = cv2.imread(img_path)
        image_target = torch.from_numpy(preprocess(image[np.newaxis,:,:,:], channel_order='BGR')).to(self.device) # torch_tensor, -1~1, RGB, BCHW
        image_target = torch.squeeze(image_target)

        target = image_target.clone()
        # target_resized = image_target_resized.clone()
        return target

    def __getitem__(self, idx):
        img_path = self.meta_csv.iloc[idx][self.key]

        image = self.load_image(img_path)
        return image

def extract_feature(model, images):
    features = model(images, output_logits=False)
    features = features.detach().cpu()
    assert features.ndim == 2 and (features.shape[1] == 2048 or features.shape[1] == 512)
    return features


def extract(csv_path,key,arcface_path,device, batch_size=64):
    dataset = BaseDataset(csv_path,key,device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    inception_model = inception.build_inception_model().to(device)
    arcface = Backbone(112, num_layers=50, mode='ir_se', drop_ratio=0.4, affine=True)
    arcface.load_state_dict(torch.load(arcface_path,map_location=device))
    arcface.to(device)
    features = []
    arc_feat = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            features.append(extract_feature(inception_model,batch))
            x = T.Resize(112)(batch)
            arc_feat.append(extract_feature(arcface,x))
    features = torch.cat(features)
    features = features.numpy()

    arc_feat = torch.cat(arc_feat)
    arc_feat = arc_feat.numpy()
    return features, arc_feat

# def compute_psnr_ssim():



# Backbone(112, num_layers=50, mode='ir_se', drop_ratio=0.4, affine=True)


