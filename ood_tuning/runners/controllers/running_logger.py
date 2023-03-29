# python3.7
"""Contains the running controller to save the running log."""

import os
import json

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Ignore TF warning.

# pylint: disable=wrong-import-position
import torch
from torch.utils.tensorboard import SummaryWriter

from ..misc import format_time
from .base_controller import BaseController
# pylint: enable=wrong-import-position
import wandb
import numpy as np

__all__ = ['RunningLogger']


class RunningLogger(BaseController):
    """Defines the running controller to save the running log.

    This controller is able to save the log message in different formats:

    (1) Text format, which will be printed on screen and saved to the log file.
    (2) JSON format, which will be saved to `{runner.work_dir}/log.json`.
    (3) Tensorboard format.

    NOTE: The controller is set to `90` priority by default and will only be
    executed on the master worker.
    """

    def __init__(self, config=None):
        config = config or dict()
        config.setdefault('priority', 90)
        config.setdefault('every_n_iters', 1)
        config.setdefault('master_only', True)
        super().__init__(config)

        self._text_format = config.get('text_format', True)
        self._log_order = config.get('log_order', None)
        self._json_format = config.get('json_format', True)
        self._json_logpath = self._json_filename = 'log.json'
        self._tensorboard_format = config.get('tensorboard_format', True)
        self.tensorboard_writer = None


        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            self.project = config.get('project', 'Out of Domain Inversion')
            self.entity = config.get('entity', 'hyun-s')
            self.reinit = config.get('reinit', True)
            self.run_name = config.get('run_name', 'Test')            
            wandb.init(project=self.project,
                       entity=self.entity, 
                       reinit=self.reinit
                       )
            wandb.run.name = self.run_name

    def setup(self, runner):
        if self._text_format:
            runner.running_stats.log_order = self._log_order
        if self._json_format:
            self._json_logpath = os.path.join(
                runner.work_dir, self._json_filename)
        if self._tensorboard_format:
            event_dir = os.path.join(runner.work_dir, 'events')
            os.makedirs(event_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=event_dir)

    def close(self, runner):
        if self._tensorboard_format:
            self.tensorboard_writer.close()

    def postprocess(self, images):
        """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
        # input : tensor, -1~1, RGB, BCHW
        # output : np.uint8, 0~255, BGR, BHWC

        images = images.detach().cpu().numpy()
        images = (images + 1.) * 255. / 2.
        images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)[:,:,:,[2,1,0]]
        images = [wandb.Image(img) for img in images]
        return images

    def execute_after_iteration(self, runner):
        # Prepare log data.
        log_data = {name: stats.get_log_value()
                    for name, stats in runner.running_stats.stats_pool.items()}
#[2023-03-29 08:59:52][INFO] Iter     30/    75, data time: 1.412s, iter time: 2.696s, run time:  1m49s, lr (discriminator): 3.0e-03, lr (generator): 3.0e-03, kimg:     0.1, lod: 0.00, minibatch:    2, g_loss: 0.690, recon_loss: 0.358, reg_loss: 0.384, inter_loss: 0.140, d_loss: 0.000, real_grad_penalty: 0.000, Gs_beta: 0.9999 (memory: 11.3G) (ETA:  2m01s)

        # Save in text format.
        msg = f'Iter {runner.iter:6d}/{runner.total_iters:6d}'
        print(type(runner.running_stats))
        msg += f', {runner.running_stats}'
        memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        msg += f' (memory: {memory:.1f}G)'
        if 'iter_time' in log_data:
            eta = log_data['iter_time'] * (runner.total_iters - runner.iter)
            msg += f' (ETA: {format_time(eta)})'
        runner.logger.info(msg)

        # Save in JSON format.
        if self._json_format:
            with open(self._json_logpath, 'a+') as f:
                json.dump(log_data, f)
                f.write('\n')

        if self.use_wandb:
            dic = {}
            for name, value in log_data.items():
                if name in ['data_time', 'iter_time', 'run_time']:
                    continue
                
                if name.startswith('loss_'):
                    dic['loss/'] = value
                elif name.startswith('lr_'):
                    dic['learning_rate/'] = value
                elif name.startswith('image'):
                    dic[name] = self.postprocess(value)
                else:
                    dic[name] = value


        # Save in Tensorboard format.
        if self._tensorboard_format:
            for name, value in log_data.items():
                if name in ['data_time', 'iter_time', 'run_time']:
                    continue
                if name.startswith('loss_'):
                    self.tensorboard_writer.add_scalar(
                        name.replace('loss_', 'loss/'), value)
                elif name.startswith('lr_'):
                    self.tensorboard_writer.add_scalar(
                        name.replace('lr_', 'learning_rate/'), value)
                else:
                    self.tensorboard_writer.add_scalar(name, value)

        # Clear running stats.
        runner.running_stats.clear()
