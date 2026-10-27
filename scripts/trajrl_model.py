"""
TrajRL model architecture (mirrored from Xfc30/TrajRL @ main, simplified).
This script is sufficient to verify the model geometry (shapes and per-block
patch hierarchy). It is NOT a faithful copy of the upstream codebase — only
the visible architecture elements needed for the smoke test.

Upstream components replaced with minimal stand-ins:
  - NodeEmbedding: simplified Linear-project version (upstream combines GAT,
    time-period, time-interval encoders); here it produces a (bs, seq_len,
    d_model) embedding through a sum + Linear fuse.
  - CrossEncoder/TransformerEncoderLayer/MultiheadAttention: imported
    implementations; geometry matches upstream.

A real reproduction run should clone https://github.com/Xfc30/TrajRL
directly (git clone) and use the upstream TrajRLContrastiveLM, NOT this file.
"""

from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ======= Layers =======

class PatchGRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, x):
        bs, num_patch, patch_len, d_model = x.size()
        x = x.reshape(-1, patch_len, d_model)
        h0 = torch.zeros(
            self.gru.num_layers * (2 if self.bidirectional else 1),
            x.size(0), self.hidden_size, device=x.device,
        )
        out, _ = self.gru(x, h0)
        out = torch.mean(out, dim=1)
        out = out.reshape(bs, num_patch,
                          self.hidden_size * (2 if self.bidirectional else 1))
        if self.bidirectional:
            out = self.linear(out)
        return out


class Patcher(nn.Module):
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, z_emb):
        # accepts [bs, seq_len, d_model] for z_emb
        bs, seq_len, d_model = z_emb.shape
        num_patch = (seq_len - self.patch_len) // self.stride + 1
        # produce [bs, num_patch, 1, patch_len, d_model]
        out = []
        for i in range(num_patch):
            s = i * self.stride
            e = s + self.patch_len
            out.append(z_emb[:, s:e, :].unsqueeze(1))  # [bs, 1, patch_len, d_model]
        out = torch.stack(out, dim=1)                  # [bs, num_patch, 1, patch_len, d_model]
        return out


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


def get_activation_fn(act):
    return nn.GELU() if act == "gelu" else nn.ReLU()


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, attn_dropout=0.,
                 proj_dropout=0., res_attention=False):
        super().__init__()
        self.n_heads = n_heads
        self.res_attention = res_attention
        self.q = nn.Linear(d_model, n_heads * d_k)
        self.k = nn.Linear(d_model, n_heads * d_k)
        self.v = nn.Linear(d_model, n_heads * d_v)
        self.proj = nn.Linear(n_heads * d_v, d_model)
        self.drop_attn = nn.Dropout(attn_dropout)
        self.drop_proj = nn.Dropout(proj_dropout)

    def forward(self, q, k, v, prev=None):
        bs = q.size(0)
        q = self.q(q).view(bs, -1, self.n_heads, q.size(-1)//self.n_heads).transpose(1, 2)
        k = self.k(k).view(bs, -1, self.n_heads, k.size(-1)//self.n_heads).transpose(1, 2)
        v = self.v(v).view(bs, -1, self.n_heads, v.size(-1)//self.n_heads).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if prev is not None:
            scores = scores + prev
        attn = scores.softmax(dim=-1)
        attn = self.drop_attn(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(
            bs, -1, self.n_heads * v.size(-1))
        out = self.drop_proj(self.proj(out))
        if self.res_attention:
            return out, attn, scores
        return out, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        d_k = d_model // n_heads
        d_v = d_model // n_heads
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v,
                                            attn_dropout, dropout, res_attention)
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm

    def forward(self, src, prev=None):
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)
        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        if self.res_attention:
            return src, scores
        return src


class CrossEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, norm='BatchNorm',
                 attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff or 4*d_model,
                                    norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                    activation=activation, res_attention=res_attention,
                                    pre_norm=pre_norm)
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores)
            return output
        for mod in self.layers:
            output = mod(output)
        return output


class Encoder(nn.Module):
    """Patch encoder: GRU on patches → Transformer on patch sequence."""

    def __init__(self, num_patch, patch_len, n_layers=3, d_model=256, n_heads=8,
                 d_ff=256, dropout=0., attn_dropout=0., act="gelu",
                 res_attention=True, pre_norm=False, pe="zeros", learn_pe=True,
                 **kwargs):
        super().__init__()
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.gru = PatchGRUEncoder(d_model, d_model, num_layers=2)
        self.W_pos = nn.Parameter(torch.zeros(1, num_patch, d_model),
                                   requires_grad=learn_pe)
        nn.init.trunc_normal_(self.W_pos, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.encoder = CrossEncoder(
            d_model, n_heads, d_ff=d_ff, norm='BatchNorm',
            attn_dropout=attn_dropout, dropout=dropout, activation=act,
            res_attention=res_attention, n_layers=n_layers, pre_norm=pre_norm,
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.gru(x)
        u = x.reshape(bs, self.num_patch, self.d_model)
        u = self.dropout(u + self.W_pos[:, :self.num_patch, :])
        z = self.encoder(u)
        z = z.reshape(-1, self.num_patch, self.d_model)
        return z


class scale_block(nn.Module):
    def __init__(self, c_in=1, fea_dim=256, patch_len=4, stride=4,
                 mask_ratio=0.2, num_patch=128, d_model=256, n_heads=8,
                 shared_embedding=True, d_ff=256, attn_dropout=0., dropout=0.,
                 act="gelu", res_attention=True, pre_norm=False, store_attn=False,
                 pe='zeros', learn_pe=True, head_dropout=0., usr_num=100,
                 with_mask=False, **kwargs):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = num_patch
        self.backbone = Encoder(
            num_patch=num_patch, patch_len=patch_len, n_layers=3,
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            attn_dropout=attn_dropout, act=act, res_attention=res_attention,
            pre_norm=pre_norm, pe=pe, learn_pe=learn_pe,
        )

    def forward(self, z_emb, x=None):
        bs, seq_len, d_model = z_emb.shape
        u = z_emb.unfold(1, self.patch_len, self.stride)
        # u: [bs, num_patch, d_model, patch_len]
        u = u.unsqueeze(2).permute(0, 1, 2, 4, 3).contiguous()  # [bs, num_patch, 1, patch_len, d_model]
        zb_patch = u.sum(dim=2)                                  # [bs, num_patch, patch_len, d_model]
        zb_patch = self.backbone(zb_patch)                       # [bs, num_patch, d_model]
        return zb_patch, None, None


class SequenceMeanPoolLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Identity() if in_dim == out_dim \
            else nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = x.mean(dim=1)
        return self.linear(out)


class WeightedAverage(nn.Module):
    def __init__(self, n, d_model):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n) / n)

    def forward(self, xs):
        stacked = torch.stack(xs, dim=0)
        w = self.w.softmax(dim=0).view(-1, 1, 1)
        return (stacked * w).sum(dim=0)


# ======= Simplified NodeEmbedding (stands in for the upstream GAT+time encoder) =======

class NodeEmbedding(nn.Module):
    """
    Minimal NodeEmbedding: returns [bs, seq_len, d_model]. Produces a
    blend of node id embedding + dummy time-period/interval embedding.

    In the upstream code this combines a GAT over the road graph with
    time-period and time-interval embeddings. The geometry — that we
    output a [bs, seq_len, d_model] tensor — is what matters for verifying
    the rest of the architecture.
    """

    def __init__(self, d_model=256, dropout=0.1, add_time_in_day=True,
                 add_day_in_week=True, add_time_interval=True,
                 max_time_scale_s=3000, time_interval_scale_list=(10, 30, 60),
                 add_pe=True, node_fea_dim=64, add_gat=True, add_hgnn=False,
                 gat_heads_per_layer=(8, 16, 1), gat_features_per_layer=(16, 16, 256),
                 gat_dropout=0.1, load_trans_prob=False, avg_last=True):
        super().__init__()
        time_dim = 0
        if add_time_in_day:
            time_dim += 64
        if add_day_in_week:
            time_dim += 64
        if add_time_interval:
            time_dim += 64 * len(time_interval_scale_list)
        if time_dim == 0:
            time_dim = 1
        self.d_model = d_model
        self.node_emb = nn.Embedding(50000, node_fea_dim)
        self.time_dim = time_dim
        self.time_proj = nn.Linear(time_dim, d_model)
        self.node_to_model = nn.Linear(node_fea_dim, d_model)
        self.fuse = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, batch_temporal_mat_list=None, graph_dict=None):
        bs, seq_len = sequence.shape
        device = sequence.device
        x_node = self.node_emb(sequence)                     # [bs, seq_len, node_fea_dim]
        x_node = self.node_to_model(x_node)                  # [bs, seq_len, d_model]
        x_time_in = torch.zeros(bs, seq_len, self.time_dim, device=device)
        x_time = self.time_proj(x_time_in)                   # [bs, seq_len, d_model]
        x = self.fuse(self.dropout(x_node + x_time))         # [bs, seq_len, d_model]
        return x


# ======= Top-level TrajRL =======

class TrajRL(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()
        self.config = config
        self.vocab_size = data_feature.get('vocab_size', 50000)
        self.usr_num = data_feature.get('usr_num', 100)
        self.node_fea_dim = data_feature.get('node_fea_dim', 64)
        self.win_size = data_feature.get('win_size', 2)
        self.mask_ratio = data_feature.get('mask_ratio', 0.2)
        self.seq_len = config.get('seq_len', 128)
        self.e_blocks = config.get('num_blocks', 4)
        self.d_model = config.get('d_model', 256)
        self.dropout = config.get('dropout', 0.1)
        self.add_gat = config.get('add_gat', True)

        self.embedding = NodeEmbedding(
            d_model=self.d_model, dropout=self.dropout,
            add_gat=self.add_gat,
            gat_heads_per_layer=config.get('gat_heads_per_layer', [8, 16, 1]),
            gat_features_per_layer=config.get('gat_features_per_layer', [16, 16, 256]),
            gat_dropout=config.get('gat_dropout', 0.1),
            time_interval_scale_list=config.get('time_interval_scales', [10, 30, 60]),
        )

        self.patchtst_layers = nn.ModuleList()
        self.patchtst_layers.append(scale_block(
            c_in=1, fea_dim=self.d_model, patch_len=1, stride=1,
            mask_ratio=self.mask_ratio, num_patch=self.seq_len,
            d_model=self.d_model, usr_num=self.usr_num, with_mask=False,
        ))
        for i in range(1, self.e_blocks):
            self.patchtst_layers.append(scale_block(
                c_in=1, fea_dim=self.d_model, patch_len=self.win_size, stride=self.win_size,
                num_patch=ceil(self.seq_len / (self.win_size ** i)),
                d_model=self.d_model, usr_num=self.usr_num,
            ))

        self.pooler_blocks = nn.ModuleList([
            SequenceMeanPoolLayer(self.d_model, self.d_model)
            for _ in range(self.e_blocks)
        ])
        self.weighted_average = WeightedAverage(self.e_blocks, self.d_model)

    def forward(self, x, temporal_mat=None, graph_dict=None, mlm=False):
        embedding_output = self.embedding(
            sequence=x, batch_temporal_mat_list=temporal_mat, graph_dict=graph_dict,
        )                                                          # [bs, seq_len, d_model]
        x_embs = []
        intput = embedding_output                                  # [bs, seq_len, d_model]
        for block in self.patchtst_layers:
            x_lay_emb, _, _ = block(intput)
            x_embs.append(x_lay_emb)
            intput = x_lay_emb.reshape(intput.size(0), -1, self.d_model)
        if mlm:
            # produce MLM logits at the finest-grained scale (block 0 output)
            mask_token = nn.Parameter(torch.zeros(self.d_model)).to(x.device)
            mask_logits = torch.matmul(x_embs[0], mask_token)  # [bs, num_patch, d_model]
            mask_logits = mask_logits.mean(dim=-1)              # [bs, num_patch]
            # produce a fake mask of shape [bs, num_patch]
            fake_mask = torch.ones_like(mask_logits).bool()
            for i in range(len(self.pooler_blocks)):
                x_embs[i] = self.pooler_blocks[i](x_embs[i])
            return self.weighted_average(x_embs), mask_logits, fake_mask
        for i in range(len(self.pooler_blocks)):
            x_embs[i] = self.pooler_blocks[i](x_embs[i])
        return self.weighted_average(x_embs)


class TrajRLContrastiveLM(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()
        self.config = config
        self.trajrl = TrajRL(config, data_feature)

    def forward(self, contra_view1, contra_view2, masked_input,
                padding_masks, batch_temporal_mat,
                padding_masks1=None, padding_masks2=None,
                batch_temporal_mat1=None, batch_temporal_mat2=None,
                graph_dict=None):
        out_view1 = self.trajrl(contra_view1,
                                temporal_mat=(batch_temporal_mat1 or batch_temporal_mat),
                                graph_dict=graph_dict, mlm=True)
        if isinstance(out_view1, tuple):
            out_view1 = out_view1[0]
        out_view2 = self.trajrl(contra_view2,
                                temporal_mat=(batch_temporal_mat2 or batch_temporal_mat),
                                graph_dict=graph_dict, mlm=True)
        if isinstance(out_view2, tuple):
            out_view2 = out_view2[0]
        out3 = self.trajrl(masked_input,
                           temporal_mat=batch_temporal_mat,
                           graph_dict=graph_dict, mlm=True)
        if isinstance(out3, tuple):
            _, a, mask = out3
        else:
            a, mask = out3, None
        return out_view1, out_view2, a, mask
