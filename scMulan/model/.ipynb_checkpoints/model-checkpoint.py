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

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))



class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

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
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
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

class Masked_BCE_logits_loss_ignore_prefix_pretrain(nn.Module): ## based on prefix len to cut prefix losses
    
    def __init__(self, vocab_size, beta=200):
        super(Masked_BCE_logits_loss_ignore_prefix_pretrain,self).__init__()
        self.vocab_size = vocab_size
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.beta = beta

    def forward(self,logits_cls,logits_exp,targets,targets_exp,xlen,x_prefix_len):
        B,T,C = logits_cls.shape
        seq_lens = xlen.unsqueeze(-1)
        prefix_lens = x_prefix_len.unsqueeze(-1)
        range_tensor = torch.arange(T).cuda().unsqueeze(0).expand(B, T)
        mask_tensor = (range_tensor < seq_lens) & (range_tensor >= prefix_lens)
        batchLogi = logits_cls[mask_tensor] # (B*L,C)
        batchTar = targets[mask_tensor].type(torch.float)  # (B*L,C) 
        batchExp = logits_exp[mask_tensor] # (B*L,C)
        batchTarExp = targets_exp[mask_tensor].type(torch.float) # (B*L)
        token_mask = batchTar !=0
        batchExp = batchExp[token_mask]
        batchTarExp = batchTarExp[token_mask]

        loss_cls = self.vocab_size * self.bce_logits_loss(batchLogi,batchTar)
        loss_exp = self.mse_loss(batchExp,batchTarExp)

            
        return loss_cls + self.beta * loss_exp, loss_cls, loss_exp

@dataclass
class GPTConfig:
    block_size: int = 1000
    vocab_size: int = 1011  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    train_mode: str = 'pretrian'
    expression_level: int = 10
    ele: int = 0


class cellGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.expression_level is not None
        self.config = config
        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
                logger.info(f"Current ele mode is: {config.ele}")

        if config.ele == 0 or 2:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
        if config.ele == 1:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wee = nn.Embedding(config.expression_level + 1, config.n_embd), # +1 for non gene tokens
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
            
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.epx_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # expr level
        self.criterion = Masked_BCE_logits_loss_ignore_prefix_pretrain(config.vocab_size)

        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters

        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.info("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        
    def get_num_params(self, non_embedding=True): ## we don't have wpe
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                idx = None,
                inputs_embeds = None,
                targets = None, 
                xlen = None,
                x_prefix_len = None,
                x_expr = None,
                y_expr = None,
                return_hidden = False,
                ):
        
        # device = idx.device
        if idx is not None:
            b, t = idx.size()
            # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            # forward the GPT model itself
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if inputs_embeds is not None:
            tok_emb = inputs_embeds
        if self.config.ele == 0:
            expr_emb = torch.zeros_like(tok_emb)
        elif self.config.ele == 1:
            # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            expr_emb = self.transformer.wee(x_expr) # expression embeddings of shape (b, t, n_embd)
        elif self.config.ele == 2:
            b,t = idx.shape
            d = self.config.n_embd
            # 创建位置编码矩阵，shape为 (b, t, d)
            expr_emb = torch.zeros((b, t, d)).cuda()
            position = x_expr.unsqueeze(-1).cuda()
            # 计算位置编码中的分母部分，shape为 (d/2)
            denominator = 10000 ** (torch.arange(0, d, 2) / d).cuda()
            # 使用向量化操作计算sin和cos
            expr_emb[:, :, 0::2] = torch.sin(torch.div(position, denominator.unsqueeze(0)))
            expr_emb[:, :, 1::2] = torch.cos(torch.div(position, denominator.unsqueeze(0)))
            
        x = self.transformer.drop(tok_emb + expr_emb)
        # x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits_cls = self.lm_head(x)
            logits_exp = self.epx_head(x)
            loss, loss_cls, loss_exp = self.criterion(logits_cls=logits_cls,logits_exp=logits_exp,targets=targets,targets_exp=y_expr,xlen=xlen,x_prefix_len=x_prefix_len)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits_cls = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits_exp = self.epx_head(x[:, [-1], :])
            loss = None
            loss_cls = None
            loss_exp = None
        if return_hidden:
            return logits_cls,logits_exp, x
        return logits_cls, logits_exp, loss, loss_cls, loss_exp


    @torch.no_grad()
    def generate_cellGenesis(self, 
                      input_ids,
                      expression_level,
                      max_new_tokens, 
                      ignore_Idx = None, 
                      top_k = None, 
                      return_dict_in_generate = False, 
                      return_hidden = False,
                      gamma = 1):

        scores = ()
        hidden_states = ()
        while True:
        # for _ in range(max_new_tokens):
            # idx_cond = input_ids[:,- self.config.block_size:]
            idx_cond = input_ids
            if return_hidden:
                logits_cls,logits_exp,hidden = self(idx = idx_cond, x_expr = expression_level, return_hidden = True)
                hidden_states += (hidden,)
            else:
                logits_cls,logits_exp,loss,_,_ = self(idx = idx_cond, x_expr = expression_level)

            logits_cls = logits_cls[:,-1,:] # (B,C)
            logits_exp = logits_exp[:,-1,:] # (B,C)

            if ignore_Idx is not None:
                # return logits, ignore_Idx
                logits_cls[:,ignore_Idx] = float('-inf')
            logits_cls[:,input_ids[0,:-1]] = float('-inf')
            if top_k is not None:
                v, _ = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)))
                logits_cls[logits_cls < v[:, [-1]]] = -float('Inf')
            
            next_token_scores = logits_cls

            # Store scores, TODO attentions and hidden_states when required
            if return_dict_in_generate:
               # FIXME if output_scores: 
                scores += (next_token_scores,)

            probs = F.softmax(logits_cls, dim=-1) #(B,C)
            # probs[:,0] = gamma*probs[:,0]
            next_tokens = torch.multinomial(probs,num_samples=1) #(B,1)
            next_token_ele = logits_exp[torch.arange(logits_exp.size(0)),next_tokens.squeeze(1)].unsqueeze(1) # (B,1)
            bin_ele_next_token = torch.clamp(torch.round(next_token_ele), 0, 10).int()
            input_ids = torch.cat((input_ids,next_tokens),dim=1)
            expression_level = torch.cat((expression_level,bin_ele_next_token),dim=1)
            # check break condition
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
            return input_ids,expression_level, hidden_states
        else:return input_ids, expression_level

@dataclass
class SampleDecoderOutput(SampleDecoderOnlyOutput):

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    expression: Optional[torch.LongTensor] = None