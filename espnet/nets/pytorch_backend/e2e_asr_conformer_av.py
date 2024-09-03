# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import numpy
import torch

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import (
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.nets_utils import MLPHead


class E2E(torch.nn.Module):
    def __init__(self, odim, args, ignore_id=-1):
        torch.nn.Module.__init__(self)

        self.encoder = Encoder(
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            zero_triu=getattr(args, "zero_triu", False),
            a_upsample_ratio=args.a_upsample_ratio,
            relu_type=getattr(args, "relu_type", "swish"),
        )

        self.aux_encoder = Encoder(
            attention_dim=args.aux_adim,
            attention_heads=args.aux_aheads,
            linear_units=args.aux_eunits,
            num_blocks=args.aux_elayers,
            input_layer=args.aux_transformer_input_layer,
            dropout_rate=args.aux_dropout_rate,
            positional_dropout_rate=args.aux_dropout_rate,
            attention_dropout_rate=args.aux_transformer_attn_dropout_rate,
            encoder_attn_layer_type=args.aux_transformer_encoder_attn_layer_type,
            macaron_style=args.aux_macaron_style,
            use_cnn_module=args.aux_use_cnn_module,
            cnn_module_kernel=args.aux_cnn_module_kernel,
            zero_triu=getattr(args, "aux_zero_triu", False),
            a_upsample_ratio=args.aux_a_upsample_ratio,
            relu_type=getattr(args, "aux_relu_type", "swish"),
        )

        self.transformer_input_layer = args.transformer_input_layer
        self.a_upsample_ratio = args.a_upsample_ratio

        self.fusion = MLPHead(
            idim=args.adim + args.aux_adim,
            hdim=args.fusion_hdim,
            odim=args.adim,
            norm=args.fusion_norm,
        )

        self.proj_decoder = None
        if args.adim != args.ddim:
            self.proj_decoder = torch.nn.Linear(args.adim, args.ddim)

        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                attention_dim=args.ddim,
                attention_heads=args.dheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
        else:
            self.decoder = None
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

    def forward(self, video, audio, video_lengths, audio_lengths, label):
        video_padding_mask = make_non_pad_mask(video_lengths).to(video.device).unsqueeze(-2)
        video_feat, _ = self.encoder(video, video_padding_mask)

        audio_lengths = torch.div(audio_lengths, 640, rounding_mode="trunc")
        audio_padding_mask = make_non_pad_mask(audio_lengths).to(video.device).unsqueeze(-2)

        audio_feat, _ = self.aux_encoder(audio, audio_padding_mask)

        x = self.fusion(torch.cat((video_feat, audio_feat), dim=-1))

        # ctc loss
        loss_ctc, ys_hat = self.ctc(x, video_lengths, label)

        if self.proj_decoder:
            x = self.proj_decoder(x)

        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, video_padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)
        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        return loss, loss_ctc, loss_att, acc
