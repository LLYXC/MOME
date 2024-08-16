# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np
from torch.nn.init import normal_
from timm.models.layers import DropPath, trunc_normal_
import math

import utils
from modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config


class TwoLayerMLP(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features, 
            out_features, 
            norm_layer, 
            norm_input=True, 
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertPooler(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = x[:, 0, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output  

'''class BEiT3AdapterForImageClassification(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3AdapterForImageClassification, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        
        self.head = TwoLayerMLP(
            in_features=embed_dim*3, 
            hidden_features=embed_dim,
            out_features=num_classes, 
            norm_layer=norm_layer, 
        )
        
        self.head.apply(self._init_weights)
        init_scale = 0.001
        if isinstance(self.head.dense1, nn.Linear):
            self.head.dense1.weight.data.mul_(init_scale)
            self.head.dense1.bias.data.mul_(init_scale)
        if isinstance(self.head.dense2, nn.Linear):
            self.head.dense2.weight.data.mul_(init_scale)
            self.head.dense2.bias.data.mul_(init_scale)
            
    def forward(self, dce, dwi, t2, **kwargs):
        beit3_output = self.beit3(dce_tokens=dce, dwi_tokens=dwi, t2_tokens=t2)
        
        x = beit3_output["encoder_out"]
        
        split_position_1, split_position_2 = beit3_output["multiway_split_position_1"], beit3_output["multiway_split_position_2"]
        
        x1, x2, x3 = torch.split(
            x,
            [split_position_1, 
             split_position_2,
             x.size(1) - split_position_1-split_position_2],
             dim=1,
        )
        
        cls_x = torch.cat((x1[:,1:,:].mean(1), x2.mean(1), x3.mean(1)), dim=-1)
        
        return self.head(cls_x)
'''

class BEiT3AdapterForImageClassification(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3AdapterForImageClassification, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.fc_norm.apply(self._init_weights)
        self.head.apply(self._init_weights)
        init_scale = 0.001
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def forward(self, dce, dwi, t2, **kwargs):
        beit3_output = self.beit3(dce_tokens=dce, dwi_tokens=dwi, t2_tokens=t2)
        x = beit3_output["encoder_out"]
        
        cls_x = self.fc_norm(x[:, 0, :])
        return self.head(cls_x)


import torch.nn as nn
import torch.nn.functional as F
class BEiT3AdapterWithAMSLossForImageClassification(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            scale,
            margin,
            reduction='none',
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3AdapterWithAMSLossForImageClassification, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.fc_norm = norm_layer(embed_dim)
        #self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes, bias=False)

        self.fc_norm.apply(self._init_weights)
        #self.head.apply(self._init_weights)
        #init_scale = 0.001
        #if isinstance(self.head, nn.Linear):
        #    self.head.weight.data.mul_(init_scale)
        #    self.head.bias.data.mul_(init_scale)

        self.scale = scale
        self.margin = margin
        self.num_classes=num_classes
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, dce, dwi, t2, labels, **kwargs):
        beit3_output = self.beit3(dce_tokens=dce, dwi_tokens=dwi, t2_tokens=t2)
        x = beit3_output["encoder_out"]
        cls_x = self.fc_norm(x[:, 0, :])
        cls_x = F.normalize(cls_x, dim=1)
        for W in self.head.parameters():
            W = F.normalize(W, dim=1)
        wf = self.head(cls_x)
        numerator = self.scale * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.margin)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * excl), dim=1)
        L = numerator - torch.log(denominator)
        
        return  wf, L

@register_model
def beit3_multimodal_adapter_base_patch16_224_imageclassification(num_classes, img_size_dce, img_size_dwi, img_size_t2, 
                                              in_chans_dce, in_chans_dwi, in_chans_t2, vis_embed_norm, pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    args.normalize_output = False
    args.num_classes=num_classes
    args.img_size_dce=img_size_dce
    args.img_size_dce=img_size_dce
    args.in_chans_dce=in_chans_dce
    args.img_size_dwi=img_size_dwi
    args.in_chans_dwi=in_chans_dwi
    args.img_size_t2=img_size_t2
    args.in_chans_t2=in_chans_t2
    args.multimodal = True
    args.vis_embed_norm = vis_embed_norm
    model = BEiT3AdapterForImageClassification(args, num_classes=num_classes, **kwargs)
    return model

@register_model
def beit3_multimodal_adapter_large_patch16_224_imageclassification(num_classes, img_size_dce, img_size_dwi, img_size_t2, 
                                              in_chans_dce, in_chans_dwi, in_chans_t2, vis_embed_norm, pretrained=False, **kwargs):
    args = _get_large_config(**kwargs)
    args.normalize_output = False
    args.img_size_dce=img_size_dce
    args.in_chans_dce=in_chans_dce
    args.img_size_dwi=img_size_dwi
    args.in_chans_dwi=in_chans_dwi
    args.img_size_t2=img_size_t2
    args.in_chans_t2=in_chans_t2
    args.multimodal = True
    args.vis_embed_norm = vis_embed_norm
    model = BEiT3AdapterForImageClassification(args, num_classes=num_classes, **kwargs)
    return model

@register_model
def beit3_multimodal_adapter_base_patch16_224_imageclassification_ams(num_classes, img_size_dce, img_size_dwi, img_size_t2, 
                                              in_chans_dce, in_chans_dwi, in_chans_t2, scale, vis_embed_norm, margin, pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    args.normalize_output = False
    args.num_classes=num_classes
    args.img_size_dce=img_size_dce
    args.in_chans_dce=in_chans_dce
    args.img_size_dwi=img_size_dwi
    args.in_chans_dwi=in_chans_dwi
    args.img_size_t2=img_size_t2
    args.in_chans_t2=in_chans_t2
    args.multimodal = True
    args.vis_embed_norm = vis_embed_norm
    model = BEiT3AdapterWithAMSLossForImageClassification(args, num_classes=num_classes, scale=scale, margin=margin, **kwargs)
    return model
