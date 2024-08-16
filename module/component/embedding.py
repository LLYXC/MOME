# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionEmbedding3D(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=(384, 256, 128),
        patch_size=16,
        in_chans=6,
        embed_dim=768,
        contain_mask_token=False,
        prepend_cls_token=False,
        norm='IN',
    ):
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (img_size[2] // patch_size[2]) * (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if norm == 'IN':
            norm1 = nn.InstanceNorm3d(64)
            norm2 = nn.InstanceNorm3d(128)
            norm3 = nn.InstanceNorm3d(embed_dim)
            print('Using IN in embedding.')
        elif norm =='BN':
            norm1 = nn.BatchNorm3d(64)
            norm2 = nn.BatchNorm3d(128)
            norm3 = nn.BatchNorm3d(embed_dim)
            print('Using BN in embedding.')
        else:
            raise NotImplementedError('Normalization should be IN or BN.')
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, 64, kernel_size=3,
                      stride=2, padding=1, bias=False),
            norm1,
            #nn.SyncBatchNorm(embed_dim),
            #nn.BatchNorm3d(64),
            #nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3,
                      stride=2, padding=1, bias=False),
            norm2,
            #nn.SyncBatchNorm(embed_dim),
            #nn.BatchNorm3d(128),
            #nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),  
            nn.Conv3d(128, embed_dim, kernel_size=3,
                      stride=2, padding=1, bias=False),
            norm3,
            #nn.SyncBatchNorm(embed_dim),
            #nn.BatchNorm3d(embed_dim),
            #nn.InstanceNorm3d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        if contain_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.mask_token = None

        if prepend_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

    def num_position_embeddings(self):
        if self.cls_token is None:
            return self.num_patches
        else:
            return self.num_patches + 1

    def forward(self, x, masked_position=None, **kwargs):
        B, C, H, W, D = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2]
        ), f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        #x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.stem(x)
                
        x = x.flatten(2).transpose(1, 2)
        batch_size, seq_len, _ = x.size()

        if masked_position is not None:
            assert self.mask_token is not None
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            w = masked_position.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                batch_size, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        return x
    
class BertEmbedding(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForSequenceClassification
        self.embed = AutoModel.from_pretrained("bert-base-chinese").embeddings
        self.embed.eval()
        
    def forward(self, x):
        input_ids, token_type_ids = x['input_ids'], x['token_type_ids']
        outputs = self.embed(input_ids, token_type_ids=token_type_ids)
        
        return outputs


class TextEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        self._fill_padding_idx_with_zero()


class PositionalEmbedding(nn.Embedding):
    def forward(
        self,
        x,
        pos_add=0,
        **kwargs,
    ):
        
        positions = (
            torch.arange(pos_add, x.size(1) + pos_add, device=x.device).long().unsqueeze(0)
        )

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
