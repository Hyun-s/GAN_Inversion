import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from models.stylegan2.model import Generator



class Generator_inter(Generator):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,):
        super().__init__(
            size=size,
            style_dim=style_dim,
            n_mlp=n_mlp,
            channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel,
            lr_mlp=lr_mlp)
    def forward(
            self,
            styles,
            return_latents=False,
            return_features=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            use_f=False,
            intermediate_out=False
    ):
    # use_f to start layer
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            styles = style_t

        # if w space
        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        # elif w+ space
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # forward
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        # # 
        if use_f != False:
            i = use_f
            for conv1, conv2, noise1, noise2, to_rgb in list(zip(
                    self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ))[i:]:
                out = conv1(out, latent[:, i], noise=noise1)
                out = conv2(out, latent[:, i + 1], noise=noise2)
                skip = to_rgb(out, latent[:, i + 2], skip)
                if intermediate_out != False:
                    if intermediate_out == i:
                        inter_out_image = skip
                    

                i += 2
        else:
            i = 1
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                    self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):
                out = conv1(out, latent[:, i], noise=noise1)
                out = conv2(out, latent[:, i + 1], noise=noise2)
                skip = to_rgb(out, latent[:, i + 2], skip)
                if intermediate_out != False:
                    if intermediate_out == i:
                        inter_out_image = skip
                    

                i += 2

            image = skip

        if return_latents:
            return image, latent
        elif return_features:
            return image, out
        elif intermediate_out != False:
            return image, inter_out_image
        else:
            return image, None