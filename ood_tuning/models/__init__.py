# python3.7
"""Collects all available models together."""
import argparse
import torch

from .model_zoo import MODEL_ZOO
from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan_discriminator import StyleGANDiscriminator
from .stylegan2_generator import StyleGAN2Generator
from .stylegan2_discriminator import StyleGAN2Discriminator
from .encoder.psp import pSp
from .encoder.encoders.psp_encoders import Encoder4Editing

__all__ = [
    'MODEL_ZOO', 'PGGANGenerator', 'PGGANDiscriminator', 'StyleGANGenerator',
    'StyleGANDiscriminator', 'StyleGAN2Generator', 'StyleGAN2Discriminator',
    'build_generator', 'build_discriminator', 'build_encoder',
    'build_model', 'parse_gan_type'
]

_GAN_TYPES_ALLOWED = ['pggan', 'stylegan', 'stylegan2']
_MODULES_ALLOWED = ['generator', 'discriminator','encoder']
_ENCODER_TYPES_ALLOWED = ['e4e']


def build_generator(gan_type, resolution, **kwargs):
    """Builds generator by GAN type.

    Args:
        gan_type: GAN type to which the generator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the generator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Generator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_discriminator(gan_type, resolution, **kwargs):
    """Builds discriminator by GAN type.

    Args:
        gan_type: GAN type to which the discriminator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Discriminator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')

def build_encoder(encoder_type, **kwargs):
    """Builds encoder.

    Args:
        encoder_type: ENCODER type to which the generator belong.
        **kwargs: Additional arguments to build the generator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if encoder_type not in _ENCODER_TYPES_ALLOWED:
        raise ValueError(f'Invalid encoder type: `{encoder_type}`!\n'
                         f'Types allowed: {_ENCODER_TYPES_ALLOWED}.')
    print('-'*50)
    print(kwargs)
    print('-'*50)
    if encoder_type == 'e4e':
        ckpt = torch.load(model_path)
        opts = argparse.Namespace(**ckpt['opts'])
        e4e = Encoder4Editing(50, 'ir_se', opts)
        e4e_dict = {k.replace('encoder.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
        e4e.load_state_dict(e4e_dict)
        e4e.eval()
        e4e = e4e.to(device)
        latent_avg = ckpt['latent_avg'].to(device)

        def add_latent_avg(model, inputs, outputs):
            return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)

        e4e.register_forward_hook(add_latent_avg)
        return e4e
    raise NotImplementedError(f'Unsupported encoder type `{encoder_type}`!')
    

def build_model(gan_type, module, resolution, **kwargs):
    """Builds a GAN module (generator/discriminator/etc).

    Args:
        gan_type: GAN type to which the model belong.
        module: GAN module to build, such as generator or discrimiantor.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `module` is not supported.
        NotImplementedError: If the `module` is not implemented.
    """
    if module not in _MODULES_ALLOWED:
        raise ValueError(f'Invalid module: `{module}`!\n'
                         f'Modules allowed: {_MODULES_ALLOWED}.')

    if module == 'generator':
        return build_generator(gan_type, resolution, **kwargs)
    if module == 'discriminator':
        return build_discriminator(gan_type, resolution, **kwargs)
    if module == 'encoder':
        return build_encoder(gan_type, **kwargs)
    raise NotImplementedError(f'Unsupported module `{module}`!')


def parse_gan_type(module):
    """Parses GAN type of a given module.

    Args:
        module: The module to parse GAN type from.

    Returns:
        A string, indicating the GAN type.

    Raises:
        ValueError: If the GAN type is unknown.
    """
    if isinstance(module, (PGGANGenerator, PGGANDiscriminator)):
        return 'pggan'
    if isinstance(module, (StyleGANGenerator, StyleGANDiscriminator)):
        return 'stylegan'
    if isinstance(module, (StyleGAN2Generator, StyleGAN2Discriminator)):
        return 'stylegan2'
    raise ValueError(f'Unable to parse GAN type from type `{type(module)}`!')
