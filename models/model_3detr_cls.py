import math

import torch
import torch.nn as nn
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes

from models.helpers import GenericMLP
from models.transformer import (MaskedTransformerEncoder, TransformerEncoder,
                                TransformerEncoderLayer)


class Model3DETRCls(nn.Module):
    """
    Encoder-only 3DETR model for shape classification.
    The implementation follows the model3_detr.py implementation, with the following modifications:
        - No decoder, and no object queries.
        - Classification head added on top of the encoder.

    There are two variants of this model depending on the encoder type:
        - vanilla: a learnable [CLS] token is prepended to the encoder input
                    sequence and then its output is fed to the classification MLP.
        - masked: final encoder features are max-pooled and fed to the classification MLP.
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        dataset_config,
        encoder_dim=256,
        use_cls_token=True,
    ):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02) # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

        # Classification head created per paper description
        self.cls_head = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=[encoder_dim],
            output_dim=dataset_config.num_classes,
            norm_fn_name="bn1d",
            activation="relu",
            dropout=0.5,
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, inputs):
        point_clouds = inputs["point_clouds"]
        xyz, features = self._break_up_pc(point_clouds)

        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        if self.use_cls_token:
            # Prepend and use [CLS] token 
            batch = pre_enc_features.shape[1]
            cls_tokens = self.cls_token.expand(1, batch, -1)
            encoder_input = torch.cat([cls_tokens, pre_enc_features], dim=0)

            _, enc_features, _ = self.encoder(encoder_input, xyz=pre_enc_xyz)
            cls_feature = enc_features[0]
        else:
            # Max pool the final features for masked encoder
            _, enc_features, _ = self.encoder(pre_enc_features, xyz=pre_enc_xyz)
            cls_feature = enc_features.max(dim=0).values

        cls_logits = self.cls_head(cls_feature)

        return {"cls_logits": cls_logits}


def build_preencoder_cls(args):
    mlp_dims = [3 * int(args.use_normals), 64, 128, args.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder_cls(args):
    encoder_layer = TransformerEncoderLayer(
        d_model=args.enc_dim,
        nhead=args.enc_nhead,
        dim_feedforward=args.enc_ffn_dim,
        dropout=args.enc_dropout,
        activation=args.enc_activation,
    )
    if args.enc_type == "vanilla":
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )
        use_cls_token = True
    elif args.enc_type == "masked":
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
        )
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
        use_cls_token = False
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder, use_cls_token


def build_3detr_cls(args, dataset_config):
    pre_encoder = build_preencoder_cls(args)
    encoder, use_cls_token = build_encoder_cls(args)
    model = Model3DETRCls(
        pre_encoder,
        encoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        use_cls_token=use_cls_token,
    )
    return model, None
