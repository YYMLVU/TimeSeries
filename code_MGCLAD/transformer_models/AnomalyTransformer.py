import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import LinearAnomalyAttention, AnomalyAttention, AttentionLayer # , FusionGraphAttention
from .embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff=None, dropout=0.1, activation="relu", output_attention=False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_layer = attention_layer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.output_attention = output_attention

    def forward(self, x, graph_embed=None):
        if self.output_attention:
            new_x, queries, keys = self.attention_layer(x, x, x, graph_embed=graph_embed)
        else:
            new_x = self.attention_layer(x, x, x, graph_embed=graph_embed)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        if self.output_attention:
            return self.norm2(x + y), queries, keys
        else:
            return self.norm2(x + y)
        

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, output_attention=False, attn_mode=0):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.output_attention = output_attention

    def forward(self, x, attn_mask=None, graph_embed=None):
        # x [B, L, D]
        queries_list = []
        keys_list = []
        if self.output_attention:
            for attn_layer in self.attn_layers:
                x, queries, keys = attn_layer(x)
                queries_list.append(queries)
                keys_list.append(keys)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, graph_embed=graph_embed)

        if self.norm is not None:
            x = self.norm(x)

        if self.output_attention:
            return x, queries_list, keys_list
        else:
            return x


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, attn_mode=0, mapping_fun='ours'):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.attn_mode = attn_mode

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        dim_per_head = d_model//n_heads
        # Encoder
        if attn_mode == 0:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            LinearAnomalyAttention(win_size, False, attention_dropout=dropout,
                                                   output_attention=self.output_attention, dim_per_head=dim_per_head,
                                                   mapping_fun=mapping_fun),
                            d_model, n_heads, output_attention=self.output_attention),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                        output_attention=self.output_attention
                    ) for _ in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model),
                output_attention=self.output_attention
            )
        # elif attn_mode == 1:
        #     self.encoder = Encoder(
        #         [
        #             EncoderLayer(
        #                 AttentionLayer(
        #                     FusionGraphAttention(win_size, enc_in, False, attention_dropout=dropout,
        #                                            output_attention=self.output_attention, dim_per_head=dim_per_head),
        #                     d_model, n_heads, output_attention=self.output_attention),
        #                 d_model,
        #                 d_ff,
        #                 dropout=dropout,
        #                 activation=activation,
        #                 output_attention=self.output_attention
        #             ) for _ in range(e_layers)
        #         ],
        #         norm_layer=torch.nn.LayerNorm(d_model),
        #         output_attention=self.output_attention
        #     )
        else:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            # LinearAnomalyAttention(win_size, False, attention_dropout=dropout,
                            #                        output_attention=self.output_attention, dim_per_head=dim_per_head),
                            AnomalyAttention(win_size, False, attention_dropout=dropout,
                                             output_attention=self.output_attention),
                            d_model, n_heads, output_attention=self.output_attention),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                        output_attention=self.output_attention
                    ) for _ in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model),
                output_attention=self.output_attention
            )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x, graph_embed=None):
        enc_out = self.embedding(x)
        if self.output_attention:
            enc_out, queries_list, keys_list = self.encoder(enc_out, graph_embed=graph_embed)
        else:
            enc_out = self.encoder(enc_out, graph_embed=graph_embed)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, queries_list, keys_list
        else:
            return enc_out  # [B, L, D]
