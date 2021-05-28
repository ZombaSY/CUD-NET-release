# -*- coding: utf-8 -*-

import torch.nn as nn

from torchvision.utils import save_image
from models import model_cud as model_cud


def initialize_weights(layer, activation='relu'):

    for module in layer.modules():
        module_name = module.__class__.__name__

        if activation in ('relu', 'leaky_relu'):
            layer_init_func = nn.init.kaiming_uniform_
        elif activation == 'tanh':
            layer_init_func = nn.init.xavier_uniform_
        else:
            raise Exception('Please specify your activation function name')

        if hasattr(module, 'weight'):
            if module_name.find('Conv2') != -1:
                layer_init_func(module.weight)
            elif module_name.find('BatchNorm') != -1:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)
            elif module_name.find('Linear') != -1:
                layer_init_func(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.1)
            else:
                # print('Cannot initialize the layer :', module_name)
                pass
        else:
            pass


class CUD_NET(nn.Module):

    def __init__(self,
                 num_points=64,
                 save_figures=False,
                 clip_threshold=False):
        super(CUD_NET, self).__init__()

        self.cud_layer = model_cud.CUD_NET(num_in_channels=3,
                                           num_out_channels=3,
                                           num_points=num_points,
                                           save_figures=save_figures,
                                           clip_threshold=clip_threshold)
        initialize_weights(self.cud_layer, activation='tanh')

    def forward(self, img, fn=None):
        img = self.cud_layer(img, fn=fn)

        return img


class CUD_Loss(nn.Module):

    def __init__(self):
        super(CUD_Loss, self).__init__()
        self.crterion_layer = model_cud.CUD_Loss()

    def forward(self, _input, _output, _target, is_identity=False):
        lab_loss, ssim_loss = self.crterion_layer(_input, _output, _target, is_identity)

        return lab_loss, ssim_loss

