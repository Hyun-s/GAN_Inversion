# python3.7
"""Defines loss functions."""

import torch
import numpy as np
import torch.nn.functional as F
from models.stylegan2_generator import UpsamplingLayer
from models.stylegan2_discriminator import DownsamplingLayer
from torch.nn import MSELoss
import lpips

__all__ = ['LogisticGANLoss']

apply_loss_scaling = lambda x: x * torch.exp(x * np.log(2.0))
undo_loss_scaling = lambda x: x * torch.exp(-x * np.log(2.0))


class LogisticGANLoss(object):
    """Contains the class to compute logistic GAN loss."""

    def __init__(self, runner, d_loss_kwargs=None, g_loss_kwargs=None):
        """Initializes with models and arguments for computing losses."""
        self.d_loss_kwargs = d_loss_kwargs or dict()
        self.g_loss_kwargs = g_loss_kwargs or dict()
        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)
        self.r2_gamma = self.d_loss_kwargs.get('r2_gamma', 0.0)

        self.inter_out = self.g_loss_kwargs.get('inter_out', 7)
        self.lambda_recon = self.g_loss_kwargs.get('lambda_recon', 0.8)
        self.lambda_mse = self.g_loss_kwargs.get('lambda_mse', 1.0)
        self.lambda_reg = self.g_loss_kwargs.get('lambda_reg', 0.5)
        self.lambda_inter = self.g_loss_kwargs.get('lambda_inter', 1.0)

        self.mse =  MSELoss()
        self.lpips_fn = lpips.LPIPS(net='vgg').cuda()
        self.lpips_fn.net.requires_grad_(False)

        runner.running_stats.add(
            f'image_rec', log_format=False, log_strategy='AVERAGE')
        runner.running_stats.add(
            f'image_origin', log_format=False, log_strategy='AVERAGE')

        runner.running_stats.add(
            f'g_loss', log_format='.3f', log_strategy='AVERAGE')
        runner.running_stats.add(
            f'recon_loss', log_format='.3f', log_strategy='AVERAGE')  
        runner.running_stats.add(
            f'reg_loss', log_format='.3f', log_strategy='AVERAGE')  
        runner.running_stats.add(
            f'inter_loss', log_format='.3f', log_strategy='AVERAGE')  
        
        runner.running_stats.add(
            f'd_loss', log_format='.3f', log_strategy='AVERAGE')
        if self.r1_gamma != 0:
            runner.running_stats.add(
                f'real_grad_penalty', log_format='.3f', log_strategy='AVERAGE')
        if self.r2_gamma != 0:
            runner.running_stats.add(
                f'fake_grad_penalty', log_format='.3f', log_strategy='AVERAGE')

    @staticmethod
    def preprocess_image(images, lod=0, **_unused_kwargs):
        """Pre-process images."""
        if lod != int(lod):
            downsampled_images = F.avg_pool2d(
                images, kernel_size=2, stride=2, padding=0)
            upsampled_images = F.interpolate(
                downsampled_images, scale_factor=2, mode='nearest')
            alpha = lod - int(lod)
            images = images * (1 - alpha) + upsampled_images * alpha
        if int(lod) == 0:
            return images
        return F.interpolate(
            images, scale_factor=(2 ** int(lod)), mode='nearest')

    @staticmethod
    def compute_grad_penalty(images, scores):
        """Computes gradient penalty."""
        image_grad = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=images,
            create_graph=True,
            retain_graph=True)[0].view(images.shape[0], -1)
        penalty = image_grad.pow(2).sum(dim=1).mean()
        return penalty

    def d_loss(self, runner, data):
        """Computes loss for discriminator."""
        G = runner.models['generator']
        D = runner.models['discriminator']
        reals = self.preprocess_image(data['image'])
        reals.requires_grad = True
        labels = data.get('label', None)

        latents = torch.randn(reals.shape[0], runner.z_space_dim).cuda()
        # print(latents.shape, reals.shape)
        latents.requires_grad = True
        # TODO: Use random labels.

        f = data['basecode'].cuda()
        wp = data['detailcode'].cuda()
        fakes = G(z=None,use_wp=wp,use_f=f,basecode_layer='x03',**runner.G_kwargs_train)['image']
        
        # fakes = G(latents, label=labels, **runner.G_kwargs_train)['image']

        real_scores = D(reals, label=labels, **runner.D_kwargs_train)
        fake_scores = D(fakes, label=labels, **runner.D_kwargs_train)

        d_loss = F.softplus(fake_scores).mean()
        d_loss += F.softplus(-real_scores).mean()
        runner.running_stats.update({'d_loss': d_loss.item()})

        real_grad_penalty = torch.zeros_like(d_loss)
        fake_grad_penalty = torch.zeros_like(d_loss)
        if self.r1_gamma:
            real_grad_penalty = self.compute_grad_penalty(reals, real_scores)
            runner.running_stats.update(
                {'real_grad_penalty': real_grad_penalty.item()})
        if self.r2_gamma:
            fake_grad_penalty = self.compute_grad_penalty(fakes, fake_scores)
            runner.running_stats.update(
                {'fake_grad_penalty': fake_grad_penalty.item()})

        return (d_loss +
                real_grad_penalty * (self.r1_gamma * 0.5) +
                fake_grad_penalty * (self.r2_gamma * 0.5))



    def intermediate_loss(self, image_k, image_out, image):
        scale_factor = image_out.shape[2] / image_k.shape[2]
        scale_factor = int(scale_factor)

        upsample = UpsamplingLayer(scale_factor=scale_factor).cuda()
        downsample = DownsamplingLayer(scale_factor=scale_factor).cuda()

        diff = image_out - upsample(image_k)
        low_image = downsample(image - diff)

        return self.mse(image_k,low_image)

    def reconstruction_loss(self, x_rec, image):
        # TODO lambda recon to args

        x_rec_resized = torch.nn.functional.interpolate(x_rec, size=(256,256), mode='bicubic')
        target_resized = torch.nn.functional.interpolate(image, size=(256,256), mode='bicubic')
        lpips_loss = torch.mean(self.lpips_fn(target_resized, x_rec_resized))
        
        mse_loss = self.mse(x_rec,image)

        return self.lambda_mse*mse_loss + self.lambda_recon*lpips_loss

    def g_loss(self, runner, data):  # pylint: disable=no-self-use
        """Computes loss for generator."""
        # TODO: Use random labels.
        G = runner.models['generator']
        D = runner.models['discriminator']
        batch_size = data['image'].shape[0]
        labels = data.get('label', None)

        # TODO params to args

        

        latents = torch.randn(batch_size, runner.z_space_dim).cuda()
        # out = G(latents, label=labels, **runner.G_kwargs_train)

        f = data['basecode'].cuda()
        wp = data['detailcode'].cuda()

        out = G(z=None,use_wp=wp,use_f=f,basecode_layer='x03',**runner.G_kwargs_train)
        
        image_origin = data['image']
        image_first_recon = data['first_recon'] # todo
        image_rec = out['image']
        inter = out[f'rgb{self.inter_out}']

        # Reconstruction Loss
        recon_loss = self.reconstruction_loss(image_rec, image_origin)

        # Regularization Loss
        if self.lambda_reg > 0:
            reg_recon_loss = self.reconstruction_loss(image_rec, image_first_recon)

            fake_scores = D(image_rec, label=labels, **runner.D_kwargs_train)
            adv_loss = F.softplus(-fake_scores).mean()
            reg_loss = reg_recon_loss + adv_loss
        else:
            reg_loss = 0
        # Intermediate Loss
        if self.lambda_inter > 0:
            inter_loss = self.intermediate_loss(inter, image_rec, image_origin)
        else:    
            inter_loss = 0
        g_loss = recon_loss + self.lambda_reg*reg_loss + self.lambda_inter*inter_loss


        runner.running_stats.update({'image_rec':image_rec.item()})
        runner.running_stats.update({'image_origin':image_origin.item()})
        runner.running_stats.update({'g_loss': g_loss.item()})
        runner.running_stats.update({'recon_loss': recon_loss.item()})
        runner.running_stats.update({'reg_loss': reg_loss.item()})
        runner.running_stats.update({'inter_loss': inter_loss.item()})

        return g_loss
