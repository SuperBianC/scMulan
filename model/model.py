from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import inspect
import math
from loguru import logger
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.generation.utils import SampleDecoderOnlyOutput


def new_gelu(x):
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Masked_BCE_logits_loss_ignore_prefix_pretrain(nn.Module):
    def __init__(self, vocab_size, beta=200):
        super(Masked_BCE_logits_loss_ignore_prefix_pretrain, self).__init__()
        self.vocab_size = vocab_size
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.beta = beta

    def forward(self, logits_cls, logits_exp, targets, targets_exp, xlen, x_prefix_len):
        B, T, C = logits_cls.shape
        seq_lens = xlen.unsqueeze(-1)
        prefix_lens = x_prefix_len.unsqueeze(-1)
        range_tensor = torch.arange(T).cuda().unsqueeze(0).expand(B, T)
        mask_tensor = (range_tensor < seq_lens) & (range_tensor >= prefix_lens)
        batchLogi = logits_cls[mask_tensor]
        batchTar = targets[mask_tensor].type(torch.float)
        batchExp = logits_exp[mask_tensor]
        batchTarExp = targets_exp[mask_tensor].type(torch.float)
        token_mask = batchTar != 0
        batchExp = batchExp[token_mask]
        batchTarExp = batchTarExp[token_mask]
        loss_cls = self.vocab_size * self.bce_logits_loss(batchLogi, batchTar)
        loss_exp = self.mse_loss(batchExp, batchTarExp)
        return loss_cls + self.beta * loss_exp, loss_cls, loss_exp


@dataclass
class MulanConfig:
    vocab_size: int = 2222
    n_layer: int = 24
    n_head: int = 20
    n_embd: int = 1120
    dropout: float = 0.0
    bias: bool = True
    expression_level: int = 10


class scMulan(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.expression_level is not None
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wee=nn.Embedding(config.expression_level + 1, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.epx_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.criterion = Masked_BCE_logits_loss_ignore_prefix_pretrain(
            config.vocab_size
        )
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx=None,
        inputs_embeds=None,
        targets=None,
        xlen=None,
        x_prefix_len=None,
        x_expr=None,
        y_expr=None,
        return_hidden=False,
    ):
        if idx is not None:
            b, t = idx.size()
            tok_emb = self.transformer.wte(idx)

        if inputs_embeds is not None:
            tok_emb = inputs_embeds
        if self.config.ele == 0:
            expr_emb = torch.zeros_like(tok_emb)
        elif self.config.ele == 1:
            expr_emb = self.transformer.wee(x_expr)
        elif self.config.ele == 2:
            b, t = idx.shape
            d = self.config.n_embd
            expr_emb = torch.zeros((b, t, d)).cuda()
            position = x_expr.unsqueeze(-1).cuda()
            denominator = 10000 ** (torch.arange(0, d, 2) / d).cuda()
            expr_emb[:, :, 0::2] = torch.sin(
                torch.div(position, denominator.unsqueeze(0))
            )
            expr_emb[:, :, 1::2] = torch.cos(
                torch.div(position, denominator.unsqueeze(0))
            )
        x = self.transformer.drop(tok_emb + expr_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits_cls = self.lm_head(x)
            logits_exp = self.epx_head(x)
            loss, loss_cls, loss_exp = self.criterion(
                logits_cls=logits_cls,
                logits_exp=logits_exp,
                targets=targets,
                targets_exp=y_expr,
                xlen=xlen,
                x_prefix_len=x_prefix_len,
            )
        else:
            logits_cls = self.lm_head(x[:, [-1], :])
            logits_exp = self.epx_head(x[:, [-1], :])
            loss = None
            loss_cls = None
            loss_exp = None
        if return_hidden:
            return logits_cls, logits_exp, x
        return logits_cls, logits_exp, loss, loss_cls, loss_exp

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        if "lm_head.weight" in decay:
            decay.remove("lm_head.weight")
            no_decay.add("lm_head.weight")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        return optimizer

    @torch.no_grad()
    def generate_cellGenesis(
        self,
        input_ids,
        expression_level,
        max_new_tokens,
        ignore_Idx=None,
        top_k=None,
        return_dict_in_generate=False,
        return_hidden=False,
    ):
        scores = ()
        hidden_states = ()
        while True:
            idx_cond = input_ids
            if return_hidden:
                logits_cls, logits_exp, hidden = self(
                    idx=idx_cond, x_expr=expression_level, return_hidden=True
                )
                hidden_states += (hidden,)
            else:
                logits_cls, logits_exp, loss, _, _ = self(
                    idx=idx_cond, x_expr=expression_level
                )

            logits_cls = logits_cls[:, -1, :]
            logits_exp = logits_exp[:, -1, :]

            if ignore_Idx is not None:
                logits_cls[:, ignore_Idx] = float("-inf")
            logits_cls[:, input_ids[0, :-1]] = float("-inf")
            if top_k is not None:
                v, _ = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)))
                logits_cls[logits_cls < v[:, [-1]]] = -float("Inf")

            next_token_scores = logits_cls

            if return_dict_in_generate:
                scores += (next_token_scores,)

            probs = F.softmax(logits_cls, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            next_token_ele = logits_exp[
                torch.arange(logits_exp.size(0)), next_tokens.squeeze(1)
            ].unsqueeze(1)
            bin_ele_next_token = torch.clamp(torch.round(next_token_ele), 0, 10).int()
            input_ids = torch.cat((input_ids, next_tokens), dim=1)
            expression_level = torch.cat((expression_level, bin_ele_next_token), dim=1)
            if next_tokens == 0 or len(input_ids[0]) >= max_new_tokens:
                break

        if return_dict_in_generate:
            return SampleDecoderOutput(
                sequences=input_ids,
                scores=scores,
                hidden_states=hidden_states,
                expression=expression_level,
            )
        elif return_hidden:
            return input_ids, expression_level, hidden_states
        else:
            return input_ids, expression_level


@dataclass
class SampleDecoderOutput(SampleDecoderOnlyOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    expression: Optional[torch.LongTensor] = None
