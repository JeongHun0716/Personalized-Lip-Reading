import sys,logging
import contextlib
from argparse import Namespace

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model, BaseFairseqModel, FairseqEncoderDecoderModel
from typing import Any, Optional
from fairseq import utils

from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import (
make_non_pad_mask,
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II, MISSING

from pathlib import Path
from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


logger = logging.getLogger(__name__)

@dataclass
class ConformerEncoderModelConfig(FairseqDataclass):
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"},
    )
    attention_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"},
    )
    attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"},
    )
    linear_units: int = field(
        default=3072, metadata={"help": "num encoder embedding dim of FFN"},
    )
    num_blocks: int = field(
        default=12, metadata={"help": "num of encoder layers"},
    )
    input_layer: str = field(
        default='conv3d', metadata={"help": "conv1d or conv3d"},
    )
    dropout_rate: float = field(
        default=0.1, metadata={"help": "dropout probaility in the encoder"},
    )
    positional_dropout_rate: float = field(
        default=0.1, metadata={"help": "dropout probaility in the input layer"},
    )
    attention_dropout_rate: float = field(
        default=0.1, metadata={"help": "dropout probailbity in the attention"},
    )
    encoder_attn_layer_type: str = field(
        default='rel_mha', metadata={"help": "encoder_attn_layer_type"},
    )
    macaron_style: bool = field(
        default=True, metadata={"help": "macaron_style"},
    )
    use_cnn_module: bool = field(
        default=True, metadata={"help": "use_cnn_module"},
    )
    cnn_module_kernel: int = field(
        default=31, metadata={"help": "cnn_module_kernel"},
    )
    zero_triu: bool = field(
        default=False, metadata={"help": "zero_triu"},
    )
    a_upsample_ratio: int = field(
        default=1, metadata={"help": "a_upsample_ratio"},
    )
    relu_type: str = field(
        default='swish', metadata={"help": "relu_type"},
    )
    
    
class ConformerEncoderWrapper(FairseqEncoder):
    def __init__(self, conformer):
        super().__init__(None)
        self.conformer = conformer
    
    def forward(self, source, padding_mask, **kwargs):
        input = source['video']

        output, _ = self.conformer(xs=input.transpose(1,2).contiguous(), masks=~padding_mask.unsqueeze(-2)) 

        return {
            "encoder_out" : output.transpose(0,1).contiguous(), # B x T x D -> T x B x D
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask
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

@dataclass
class ConformerLLMConfig(ConformerEncoderModelConfig):
    decoder_embed_dim: int = field(
        default=4096, metadata={"help": "encoder embedding dimension"}
    )
    adpater_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    llm_ckpt_path: str = field(
        default=MISSING
    )
    conformer_ckpt_path: str = field(
        default=MISSING, metadata={"help": "path to confomrmer checkpoint"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
      
        
@register_model("conformer_llm", dataclass=ConformerLLMConfig)
class ConformerLLM(BaseFairseqModel):
    def __init__(self, encoder,  decoder, tokenizer, cfg):
        super().__init__() 
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder   
        self.tokenizer = tokenizer
        self.vfeat_to_llm = nn.Linear(cfg.attention_dim, 4096)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.freeze_params = [n for n,p in self.named_parameters() if p.requires_grad == False]
        
        
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        conformer_ = Encoder(
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
        src_pth = os.path.dirname(os.path.realpath(__file__))
        ckpt_path = f'{src_pth}/pretrained_models/conformer_encoder/pretrained_lrs3/vsr_trlrs3_base.pth'
        
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        tmp_ckpt = {k[8:]: v for k, v in ckpt.items() if 'encoder' in k }
        conformer_.load_state_dict(tmp_ckpt)
        
        conformer = ConformerEncoderWrapper(conformer_)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        llm_pth = f'{src_pth}/pretrained_models/llm/Meta-Llama-3-8B'
        llm = AutoModelForCausalLM.from_pretrained(llm_pth, quantization_config=bnb_config)
        
        for param in llm.parameters():
            param.requires_grad = False

        tokenizer = AutoTokenizer.from_pretrained(llm_pth)

        return ConformerLLM(conformer, llm, tokenizer, cfg)
    

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
        B,T = kwargs['padding_mask'].size()
        output = self.encoder(**kwargs)
        output['encoder_out'] = output['encoder_out'].transpose(0,1)
        output['encoder_out'] = self.vfeat_to_llm(output['encoder_out'])
        
        
        B, T, D = output['encoder_out'].size()
        instruction = kwargs['source']['text']

        instruction_embedding = self.decoder.model.embed_tokens(instruction)
  
        labels = kwargs['target_list'].clone()
        labels_embedding = self.decoder.model.embed_tokens(labels)
        labels = labels.masked_fill(labels == 0, -100)

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

        output = self.encoder(**kwargs)
        output['encoder_out'] = self.vfeat_to_llm(output['encoder_out']).transpose(0,1)

        B, T, D = output['encoder_out'].size()
        
        instruction = kwargs['source']['text']
        instruction_embedding = self.decoder.model.embed_tokens(instruction)
        llm_input = torch.cat((instruction_embedding, output['encoder_out']), dim=1)
        
        self.decoder.generation_config.pad_token_id = self.decoder.generation_config.eos_token_id
        outputs = self.decoder.generate(inputs_embeds=llm_input, num_beams=1,  max_new_tokens=max_length, min_length=min_length)

        return outputs
            
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
