# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn

from .encoder import Encoder
from .encoder_aim import Encoder_Aim
from . component.embedding import (
    PositionalEmbedding,
    TextEmbedding,
    BertEmbedding,
    VisionEmbedding3D,
)
from .component.multiway_network import MutliwayEmbedding

#TODO modify the embedding layers

class BEiT3(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        assert args.multiway
        assert args.vocab_size > 0
        assert not args.share_encoder_input_output_embed
        #self.text_embed = TextEmbedding(args.vocab_size, args.encoder_embed_dim)
        #self.text_embed = BertEmbedding()
        #self.text_linear = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        
        self.dce_embed = VisionEmbedding3D(
            args.img_size_dce,
            args.patch_size,
            args.in_chans_dce,
            args.encoder_embed_dim,
            norm=args.vis_embed_norm,
            contain_mask_token=False,
            prepend_cls_token=True,
        )
        
        self.dwi_embed = VisionEmbedding3D(
            args.img_size_dwi,
            args.patch_size,
            args.in_chans_dwi,
            args.encoder_embed_dim,
            norm=args.vis_embed_norm,
            contain_mask_token=False,
            prepend_cls_token=False,
        )
        
        self.t2_embed = VisionEmbedding3D(
            args.img_size_t2,
            args.patch_size,
            args.in_chans_t2,
            args.encoder_embed_dim,
            norm=args.vis_embed_norm,
            contain_mask_token=False,
            prepend_cls_token=False,
        )

        # being consistent with Fairseq, which starts from 2 for position embedding
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(self.dce_embed.num_position_embeddings() + 1, args.encoder_embed_dim),
                PositionalEmbedding(self.dwi_embed.num_position_embeddings(), args.encoder_embed_dim),
                PositionalEmbedding(self.t2_embed.num_position_embeddings(), args.encoder_embed_dim),
            ],
            dim=1,
        )
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
        )
        
        # self.encoder = Encoder_Aim(
        #     args,
        #     embed_tokens=None,
        #     embed_positions=embed_positions,
        #     output_projection=None,
        #     is_encoder_decoder=False,
        # )

    def forward(
        self,
        dce_tokens=None,
        dwi_tokens=None,
        t2_tokens=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
    ):
        assert dce_tokens is not None or dwi_tokens is not None or t2_tokens is not None 

        x1 = self.dce_embed(dce_tokens, vision_masked_position)
        multiway_split_position_1 = x1.size(1)
        x2 = self.dwi_embed(dwi_tokens, vision_masked_position)
        multiway_split_position_2 = x2.size(1)
        x3 = self.t2_embed(t2_tokens, vision_masked_position)
        x = torch.cat([x1, x2, x3], dim=1)

        encoder_out = self.encoder(
            src_tokens=None,
            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position_1=multiway_split_position_1,
            multiway_split_position_2=multiway_split_position_2,
            incremental_state=incremental_state,
            positions=positions,
        )
        encoder_out["multiway_split_position_1"] = multiway_split_position_1
        encoder_out["multiway_split_position_2"] = multiway_split_position_2

        return encoder_out
