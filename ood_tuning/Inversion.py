from easydict import EasyDict
import os, inspect, shutil, json, sys, shutil
import argparse
import subprocess
import csv
import cv2
import numpy as np
import random
import lpips
import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm


sys.path.append('/content/GAN_Inversion/ood_tuning/BDInvert')

from models.stylegan2_generator import StyleGAN2Generator
from models.stylegan_basecode_encoder import encoder_simple

from image_tools import preprocess, postprocess, Lanczos_resizing
from models.stylegan_basecode_encoder import encoder_simple
from pca_p_space import project_w2pN
from collections import defaultdict

class BDInvert():
    def __init__(self, device, args):
        self.args = args
        self.device = device
        self.generator = self.set_generator().to(self.device)
        self.basecode_encoder = self.set_basecode_encoder()
        self.fix_seed()
        self.p_mean_latent, self.p_eigen_values, self.p_eigen_vectors = self.set_Pnorm()
    def fix_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        print('fixed seed',self.args.seed)


    def set_generator(self):
        g = torch.load(self.args.generator_pth_path)['generator']
        generator = StyleGAN2Generator(resolution=1024)
        generator.load_state_dict(g)
        print('StyleGAN2 keys matched successfully')
        return generator

    def set_basecode_encoder(self):
        _,_,_,basecode = self.init_codes()
        self.encoder_input_shape = [3, basecode.shape[2]*8, basecode.shape[3]*8]
        self.encoder_output_shape = basecode.shape[1:]
        basecode_encoder = encoder_simple(encoder_input_shape=self.encoder_input_shape,
                                      encoder_output_shape=self.encoder_output_shape,
                                      cfg=self.args.encoder_cfg)
        basecode_encoder.load_state_dict(torch.load(self.args.encoder_pt_path)['encoder'])
        basecode_encoder.to(self.device).eval()
        basecode_encoder.requires_grad_(False)
        return basecode_encoder

    def set_Pnorm(self):
        p_mean_latent = np.load(f'{self.args.pnorm_root}/mean_latent.npy')
        p_eigen_values = np.load(f'{self.args.pnorm_root}/eigen_values.npy')
        p_eigen_vectors = np.load(f'{self.args.pnorm_root}/eigen_vectors.npy')

        p_mean_latent = torch.from_numpy(p_mean_latent).to(self.device)
        p_eigen_values = torch.from_numpy(p_eigen_values).to(self.device)
        p_eigen_vectors = torch.from_numpy(p_eigen_vectors).to(self.device)
        return p_mean_latent, p_eigen_values, p_eigen_vectors

    def init_codes(self):
        basecode_layer = int(np.log2(self.args.basecode_spatial_size) - 2) * 2
        basecode_layer = f'x{basecode_layer-1:02d}'
        with torch.no_grad():
            z = torch.randn(1, 512).to(self.device)
            w = self.generator.mapping(z, label=None)['w']
            wp = self.generator.truncation(w, trunc_psi=self.args.trunc_psi, trunc_layers=self.args.trunc_layers)
            basecode = self.generator.synthesis(wp, randomize_noise=self.args.randomize_noise)[basecode_layer]
        return z,w,wp,basecode

    def load_image(self,img_path):
        image = cv2.imread(img_path)
        image_target = torch.from_numpy(preprocess(image[np.newaxis, :], channel_order='BGR')).cuda() # torch_tensor, -1~1, RGB, BCHW
        image_target = Lanczos_resizing(image_target, (self.generator.resolution,self.generator.resolution))
        image_target_resized = Lanczos_resizing(image_target, (256,256))

        target = image_target.clone()
        target_resized = image_target_resized.clone()
        return target, target_resized


    def invert(self,
               image_paths):
        # target,target_resize = self.load_image(path)
        save_csv = []
        if not self.save_dir:
            save_image_path = os.path.join(self.save_dir,'Image_o')
            save_F_path = os.path.join(self.save_dir,'F')
            save_w_m_plus_path = os.path.join(self.save_dir,'w_m_plus')
            
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(save_image_path, exist_ok=True)
            os.makedirs(save_F_path, exist_ok=True)
            os.makedirs(save_w_m_plus_path, exist_ok=True)


        lpips_fn = lpips.LPIPS(net='vgg').cuda()
        lpips_fn.net.requires_grad_(False)
        dic = defaultdict(dict)
        for idx, path in enumerate(image_paths):
            image_id = os.path.split(path)[-1]
            target,target_resized = self.load_image(path)
            # Generate starting detail codes
            detailcode_starting = self.generator.truncation.w_avg.clone().detach()
            detailcode_starting = detailcode_starting.view(1, 1, -1)
            detailcode_starting = detailcode_starting.repeat(1, self.generator.num_layers, 1)
            detailcode = detailcode_starting.clone()
            detailcode.requires_grad_(True)

            # Define starting base code
            basecode_layer = int(np.log2(self.args.basecode_spatial_size) - 2) * 2
            basecode_layer = f'x{basecode_layer-1:02d}'
            if basecode_layer is not None:
                with torch.no_grad():
                    encoder_input = Lanczos_resizing(target, (self.encoder_input_shape[1], self.encoder_input_shape[2]))
                    basecode_starting = self.basecode_encoder(encoder_input)
                    basecode = basecode_starting.clone()
                basecode.requires_grad_(True)
            

            # Define optimizer
            optimizing_variable = []
            optimizing_variable.append(detailcode)
            if basecode_layer is not None:
                optimizing_variable.append(basecode)
            optimizer = torch.optim.Adam(optimizing_variable, lr=self.args.lr)
            
            for iter in tqdm(range(self.args.num_iters)):
                loss = 0.
                x_rec = self.generator.synthesis(detailcode, randomize_noise=self.args.randomize_noise,
                                            basecode_layer=basecode_layer, basecode=basecode)['image']

                # MSE
                mse_loss = torch.mean((x_rec-target)**2)
                loss += mse_loss

                # LPIPS
                x_rec_resized = torch.nn.functional.interpolate(x_rec, size=(256,256), mode='bicubic')
                lpips_loss = torch.mean(lpips_fn(target_resized, x_rec_resized))
                loss += lpips_loss * self.args.weight_perceptual_term

                # Base code regularization
                reg_basecode_loss = torch.mean((basecode-basecode_starting)**2)
                loss += reg_basecode_loss * self.args.weight_basecode_term

                # Detail code regularization
                if self.args.weight_pnorm_term:
                    pprojected_detailcode = project_w2pN(detailcode[0], 
                                                        self.p_mean_latent, 
                                                        self.p_eigen_values, 
                                                        self.p_eigen_vectors)
                    reg_detailcode_loss = torch.mean((pprojected_detailcode)**2)
                    loss += reg_detailcode_loss * self.args.weight_pnorm_term

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            with torch.no_grad():
                x_rec = self.generator.synthesis(detailcode, randomize_noise=self.args.randomize_noise,
                                            basecode_layer=basecode_layer, basecode=basecode)['image']
                rec_image = postprocess(x_rec.clone())[0]
                basecode_save = basecode.clone().detach().cpu().numpy()
                detailcode_save = detailcode.clone().detach().cpu().numpy()

            dic[image_id] ={
                'x_rec':target,
                'rec_image':rec_image,
                'basecode':basecode_save,
                'detailcode':detailcode_save
            }
            if self.save_dir != False:
                file_id = os.path.splitext(image_id)[0]
                print(str(idx)+ ' : '+image_id+' saving')
                save_path = os.path.join(self.save_dir,file_id)
                os.makedirs(save_path, exist_ok=True)
                tup = {'Image_path': os.path.join(save_path,image_id),
                       'Image_rec_path': os.path.join(save_path,file_id+'_rec.png'),
                       'f_path': os.path.join(save_path,file_id+'_f.npy'),
                       'w_m_plus_path': os.path.join(save_path,file_id+'_w_m_plus.npy')}
                

                shutil.copy(path, tup['Image_path'])
                cv2.imwrite(tup['Image_rec_path'],rec_image)
                np.save(tup['f_path'], basecode_save)
                np.save(tup['w_m_plus_path'], detailcode_save)
                
                save_csv.append(tup)

                # origin image(path), image_o f, wm+

        df = pd.DataFrame(save_csv)
        df.to_csv(os.path.join(self.save_dir,'paths.csv'), index=False)
        return df
