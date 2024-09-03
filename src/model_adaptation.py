# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#=================== 202 ======================

import sys,logging
import contextlib
import tempfile
from argparse import Namespace
from typing import Any, Optional, Tuple
from peft import LoraConfig, get_peft_model
import os

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, FairseqEncoderDecoderModel, register_model
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder


from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING, open_dict


from espnet.nets.pytorch_backend.transformer.encoder_udp import Encoder_udp
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import (
make_non_pad_mask,
)

from src.model_baseline import ConformerLLMConfig, ConformerEncoderWrapper


DBG=True if len(sys.argv) == 1 else False
EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
import pdb


logger = logging.getLogger(__name__)


@dataclass
class Vision_Language_Adaptation(ConformerLLMConfig):
    prompt_length: int = field(
        default=10
    )
    target_speaker_padding_prompt_pth: str = field(
        default='eval'
    )
    speaker_id: str = field(
        default='00000'
    )

class AdaptConformerEncoderWrapper(FairseqEncoder):
    def __init__(self, conformer):
        super().__init__(None)
        self.conformer = conformer
        
    def forward(self, source, padding_mask, udps, add_prompt=None, concat_prompt=None, **kwargs):
        input = source['video']
        if add_prompt != None:
            input = input + add_prompt
        lengths = (padding_mask == False).sum(dim=1)
        if concat_prompt != None:
            B,T,D = concat_prompt.size()
            lengths = lengths + T
        padding_mask = make_non_pad_mask(lengths).to(padding_mask.device).unsqueeze(-2)
        output, _ = self.conformer(xs=input.transpose(1,2).contiguous(), masks=padding_mask, prompt=concat_prompt, udps=udps) 

        return {
            "encoder_out" : output.transpose(0,1).contiguous(), # B x T x D -> T x B x D
            "encoder_padding_mask": ~padding_mask.squeeze(1),  # B x T
            "padding_mask": ~padding_mask.squeeze(1)
        }
        
        
    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    
###############################
### Vision Level Adaptation ###    
###############################


@register_model("vision_adaptation", dataclass=ConformerLLMConfig)
class vision_adaptation(BaseFairseqModel):
    def __init__(self, encoder,  decoder, tokenizer, udps, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder   
        self.tokenizer = tokenizer
        self.vfeat_to_llm = nn.Linear(cfg.attention_dim, 4096)
        
        ckpt = torch.load(cfg.conformer_ckpt_path)
        self.load_state_dict(ckpt["model"], strict=False)
        
        for param in self.vfeat_to_llm.parameters():
            param.requires_grad = False


        config = LoraConfig(
                r=8, 
                lora_alpha=16, 
                target_modules=["conv1", "linear_q", "linear_k", "linear_v"], 
                lora_dropout=0.05, 
                bias="none", 
                task_type=" SEQ_2_SEQ_LM" 
            )
        self.encoder = get_peft_model(self.encoder, config)
        self.encoder.print_trainable_parameters()
        
        self.freeze_finetune_updates = cfg.freeze_finetune_updates

        self.udps = udps

        for param in self.udps.parameters():
            param.requires_grad = True
            
        self.freeze_params = [n for n,p in self.named_parameters() if p.requires_grad == False]
        
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        conformer_ = Encoder_udp(
            attention_dim = cfg.attention_dim,
            attention_heads = cfg.attention_heads,
            linear_units = cfg.linear_units,
            num_blocks = cfg.num_blocks,
            input_layer = cfg.input_layer,
            dropout_rate = cfg.dropout_rate,
            positional_dropout_rate = cfg.positional_dropout_rate,
            attention_dropout_rate = cfg.attention_dropout_rate,
            encoder_attn_layer_type = cfg.encoder_attn_layer_type,
            macaron_style = cfg.macaron_style,
            use_cnn_module= cfg.use_cnn_module,
            cnn_module_kernel = cfg.cnn_module_kernel,
            zero_triu = cfg.zero_triu,
            a_upsample_ratio = cfg.a_upsample_ratio,
            relu_type = cfg.relu_type,
        )
            
        conformer = AdaptConformerEncoderWrapper(conformer_)
        for name, param in conformer.named_parameters():
            param.requires_grad = False


        def udp_gen(feat_size, pad, channel):
            return nn.Parameter(torch.zeros([(pad * feat_size * 4 + pad * pad * 4) * channel]))

        udps = nn.ParameterDict({'udp0': udp_gen(feat_size=88, pad=3, channel=1),
           'udp1': udp_gen(feat_size=22, pad=1, channel=64),
           'udp2': udp_gen(feat_size=22, pad=1, channel=64),
           'udp3': udp_gen(feat_size=22, pad=1, channel=64),
           'udp4': udp_gen(feat_size=22, pad=1, channel=64),
           'udp5': udp_gen(feat_size=22, pad=1, channel=64),
           'udp6': udp_gen(feat_size=11, pad=1, channel=128),
           'udp7': udp_gen(feat_size=11, pad=1, channel=128),
           'udp8': udp_gen(feat_size=11, pad=1, channel=128),
           'udp9': udp_gen(feat_size=11, pad=1, channel=128),
           'udp10': udp_gen(feat_size=6, pad=1, channel=256),
           'udp11': udp_gen(feat_size=6, pad=1, channel=256),
           'udp12': udp_gen(feat_size=6, pad=1, channel=256),
           'udp13': udp_gen(feat_size=6, pad=1, channel=256),
           'udp14': udp_gen(feat_size=3, pad=1, channel=512),
           'udp15': udp_gen(feat_size=3, pad=1, channel=512),
           'udp16': udp_gen(feat_size=3, pad=1, channel=512),
           })
        

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        
        src_pth = os.path.dirname(os.path.realpath(__file__))
        llm_pth = f'{src_pth}/pretrained_models/llm/Meta-Llama-3-8B'
        llm = AutoModelForCausalLM.from_pretrained(llm_pth, quantization_config=bnb_config)
        
        
        for name, param in llm.named_parameters():
            param.requires_grad = False
            
        tokenizer = AutoTokenizer.from_pretrained(llm_pth)

        return cls(conformer, llm, tokenizer, udps, cfg)
    
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        
    def state_dict(self):
        old_state = super().state_dict()
        state = {k:v for k,v in old_state.items() if k not in self.freeze_params}
        return state
    
    def load_state_dict(self,state,**kwargs):
        # import pdb;pdb.set_trace()
        super().load_state_dict(state, strict=False)

    def forward(self, **kwargs):
         ### Visual Level Adaptation ###
        B,T = kwargs['padding_mask'].size()

        output = self.encoder(add_prompt=None, udps=self.udps, concat_prompt=None, **kwargs)
        output['encoder_out'] = output['encoder_out'].transpose(0,1)
        output['encoder_out'] = self.vfeat_to_llm(output['encoder_out'])
        #################################
        
        B, T, D = output['encoder_out'].size()
        instruction = kwargs['source']['text']

        instruction_embedding = self.decoder.model.embed_tokens(instruction)
  
        labels = kwargs['target_list'].clone()
        labels_embedding = self.decoder.model.embed_tokens(labels)

        llm_input = torch.cat((instruction_embedding, output['encoder_out'], labels_embedding), dim=1)


        _, instruction_embedding_t, _ = instruction_embedding.size()
        target_ids = torch.full((B, T + instruction_embedding_t),-100).long().to(labels.device)
        llm_labels = torch.cat((target_ids, labels), dim=1)  

        
        llm_out = self.decoder(inputs_embeds=llm_input, labels=llm_labels, return_dict=True)
        
        return llm_out.loss, llm_out.logits.to(output['encoder_out'].device)


    @torch.no_grad()
    def generate(self,
                num_beams=1,
                max_length=100,
                min_length=1,
                top_p=0.7,
                repetition_penalty=1.0,
                length_penalty=0.0,
                  **kwargs,
                ):

        B,T = kwargs['padding_mask'].size()
        output = self.encoder(add_prompt=None, udps=self.udps, concat_prompt=None, **kwargs)
        output['encoder_out'] = self.vfeat_to_llm(output['encoder_out']).transpose(0,1)


        B, T, D = output['encoder_out'].size()
        
        instruction = kwargs['source']['text']
        instruction_embedding = self.decoder.model.embed_tokens(instruction)
        llm_input = torch.cat((instruction_embedding, output['encoder_out']), dim=1)
        
        #self.decoder.config.use_cache = True
        self.decoder.generation_config.pad_token_id = self.decoder.generation_config.eos_token_id
        outputs = self.decoder.generate(inputs_embeds=llm_input, num_beams=1,  max_new_tokens=max_length, min_length=min_length)

        return outputs
    
    
##########################################
### Vision & Language Level Adaptation ###    
##########################################

@register_model("vision_language_adaptation", dataclass=Vision_Language_Adaptation)
class vision_language_adaptation(BaseFairseqModel):
    def __init__(self, encoder,  decoder, tokenizer, udps, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder   
        self.tokenizer = tokenizer
        self.vfeat_to_llm = nn.Linear(cfg.attention_dim, 4096)
        
        ckpt = torch.load(cfg.conformer_ckpt_path)
        self.load_state_dict(ckpt["model"], strict=False)
        
        for param in self.vfeat_to_llm.parameters():
            param.requires_grad = False

        config = LoraConfig(
                r=8, 
                lora_alpha=16, 
                target_modules=["conv1", "linear_q", "linear_k", "linear_v"], 
                lora_dropout=0.05, 
                bias="none", 
                task_type=" SEQ_2_SEQ_LM" 
            )
        self.encoder = get_peft_model(self.encoder, config)
        self.encoder.print_trainable_parameters()
        
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.udps = udps
        if cfg.target_speaker_padding_prompt_pth == 'eval':
            src_pth = os.path.dirname(os.path.realpath(__file__))
            target_speaker_padding_prompt_pth = f'{src_pth}/pretrained_models/adapted_model/vision/voxlrs-{cfg.speaker_id}/checkpoints/checkpoint_best.pt'
            ckpt = torch.load(target_speaker_padding_prompt_pth)["model"]
            self.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(cfg.target_speaker_padding_prompt_pth)["model"]
            self.load_state_dict(ckpt, strict=False)
            
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.udps.parameters():
            param.requires_grad = False
        
        self.prompt = nn.Parameter(torch.randn(1, cfg.prompt_length, 4096)) # B x L x D
        self.freeze_params = [n for n,p in self.named_parameters() if p.requires_grad == False]
        
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        conformer_ = Encoder_udp(
            attention_dim = cfg.attention_dim,
            attention_heads = cfg.attention_heads,
            linear_units = cfg.linear_units,
            num_blocks = cfg.num_blocks,
            input_layer = cfg.input_layer,
            dropout_rate = cfg.dropout_rate,
            positional_dropout_rate = cfg.positional_dropout_rate,
            attention_dropout_rate = cfg.attention_dropout_rate,
            encoder_attn_layer_type = cfg.encoder_attn_layer_type,
            macaron_style = cfg.macaron_style,
            use_cnn_module= cfg.use_cnn_module,
            cnn_module_kernel = cfg.cnn_module_kernel,
            zero_triu = cfg.zero_triu,
            a_upsample_ratio = cfg.a_upsample_ratio,
            relu_type = cfg.relu_type,
        )
            
        conformer = AdaptConformerEncoderWrapper(conformer_)
        for name, param in conformer.named_parameters():
            param.requires_grad = False


        def udp_gen(feat_size, pad, channel):
            return nn.Parameter(torch.zeros([(pad * feat_size * 4 + pad * pad * 4) * channel]))

        udps = nn.ParameterDict({'udp0': udp_gen(feat_size=88, pad=3, channel=1),
           'udp1': udp_gen(feat_size=22, pad=1, channel=64),
           'udp2': udp_gen(feat_size=22, pad=1, channel=64),
           'udp3': udp_gen(feat_size=22, pad=1, channel=64),
           'udp4': udp_gen(feat_size=22, pad=1, channel=64),
           'udp5': udp_gen(feat_size=22, pad=1, channel=64),
           'udp6': udp_gen(feat_size=11, pad=1, channel=128),
           'udp7': udp_gen(feat_size=11, pad=1, channel=128),
           'udp8': udp_gen(feat_size=11, pad=1, channel=128),
           'udp9': udp_gen(feat_size=11, pad=1, channel=128),
           'udp10': udp_gen(feat_size=6, pad=1, channel=256),
           'udp11': udp_gen(feat_size=6, pad=1, channel=256),
           'udp12': udp_gen(feat_size=6, pad=1, channel=256),
           'udp13': udp_gen(feat_size=6, pad=1, channel=256),
           'udp14': udp_gen(feat_size=3, pad=1, channel=512),
           'udp15': udp_gen(feat_size=3, pad=1, channel=512),
           'udp16': udp_gen(feat_size=3, pad=1, channel=512),
           })
        

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    
        src_pth = os.path.dirname(os.path.realpath(__file__))
        llm_pth = f'{src_pth}/pretrained_models/llm/Meta-Llama-3-8B'
        llm = AutoModelForCausalLM.from_pretrained(llm_pth, quantization_config=bnb_config)
        for name, param in llm.named_parameters():
            param.requires_grad = False
            
        tokenizer = AutoTokenizer.from_pretrained(llm_pth)

        return cls(conformer, llm, tokenizer, udps, cfg)
    

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        
    def state_dict(self):
        old_state = super().state_dict()
        state = {k:v for k,v in old_state.items() if k not in self.freeze_params}
        return state
    
    def load_state_dict(self,state,**kwargs):
        # import pdb;pdb.set_trace()
        super().load_state_dict(state, strict=False)

    def forward(self, **kwargs):
         ### Visual Level Adaptation ###
        B,T = kwargs['padding_mask'].size()

        output = self.encoder(add_prompt=None, udps=self.udps, concat_prompt=None, **kwargs)
        output['encoder_out'] = output['encoder_out'].transpose(0,1)
        output['encoder_out'] = self.vfeat_to_llm(output['encoder_out'])
        #################################
        
        B, T, D = output['encoder_out'].size()
        instruction = kwargs['source']['text']

        instruction_embedding = self.decoder.model.embed_tokens(instruction)
  
        labels = kwargs['target_list'].clone()
        labels_embedding = self.decoder.model.embed_tokens(labels)

        llm_input = torch.cat((instruction_embedding, self.prompt, output['encoder_out'], labels_embedding), dim=1)


        _, instruction_embedding_t, _ = instruction_embedding.size()
        target_ids = torch.full((B, T + instruction_embedding_t+10),-100).long().to(labels.device)
        llm_labels = torch.cat((target_ids, labels), dim=1)  

        
        llm_out = self.decoder(inputs_embeds=llm_input, labels=llm_labels, return_dict=True)
        
        return llm_out.loss, llm_out.logits.to(output['encoder_out'].device)


    @torch.no_grad()
    def generate(self,
                num_beams=1,
                max_length=100,
                min_length=1,
                top_p=0.7,
                repetition_penalty=1.0,
                length_penalty=0.0,
                  **kwargs,
                ):

        B,T = kwargs['padding_mask'].size()
        output = self.encoder(add_prompt=None, udps=self.udps, concat_prompt=None, **kwargs)
        output['encoder_out'] = self.vfeat_to_llm(output['encoder_out']).transpose(0,1)


        B, T, D = output['encoder_out'].size()
        
        instruction = kwargs['source']['text']
        instruction_embedding = self.decoder.model.embed_tokens(instruction)
        llm_input = torch.cat((instruction_embedding, self.prompt.cuda(), output['encoder_out']), dim=1)
        
        #self.decoder.config.use_cache = True
        self.decoder.generation_config.pad_token_id = self.decoder.generation_config.eos_token_id
        outputs = self.decoder.generate(inputs_embeds=llm_input, num_beams=1,  max_new_tokens=max_length, min_length=min_length)

        return outputs
    
    
