import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Adapted from https://github.com/AIRMEC/im4MEC

class Attn_Net_Gated(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, p_dropout_atn=0.25, n_classes=1):
        super(Attn_Net_Gated, self).__init__()

        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(p_dropout_atn))
            self.attention_b.append(nn.Dropout(p_dropout_atn))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A


class Im4MEC(nn.Module):
    def __init__(
        self,
        input_feature_size=1024,
        precompression_layer=True,
        feature_size_comp = 512,
        feature_size_attn = 256,
        dropout=True,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
        n_classes=4,
    ):
        super(Im4MEC, self).__init__()

        self.n_classes = n_classes

        if precompression_layer:
            self.compression_layer = nn.Sequential(*[
                                                    nn.Linear(input_feature_size, feature_size_comp*4), 
                                                    nn.ReLU(), 
                                                    nn.Dropout(p_dropout_fc),
                                                    nn.Linear(feature_size_comp*4, feature_size_comp*2), 
                                                    nn.ReLU(), 
                                                    nn.Dropout(p_dropout_fc),
                                                    nn.Linear(feature_size_comp*2, feature_size_comp), 
                                                    nn.ReLU(), 
                                                    nn.Dropout(p_dropout_fc)])

            dim_post_compression = feature_size_comp
        else:
            self.compression_layer = nn.Identity()
            dim_post_compression = input_feature_size

        self.attention_net = Attn_Net_Gated(
            L=dim_post_compression,
            D=feature_size_attn,
            dropout=dropout,
            p_dropout_atn=p_dropout_atn,
            n_classes=self.n_classes)

        # Classification head.
        self.classifiers = nn.ModuleList(
            [nn.Linear(dim_post_compression, 1) for i in range(self.n_classes)]
        )

        # Init weights.
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward_attention(self, h):
        A_ = self.attention_net(h)  # h shape is N_tilesxdim
        A_raw = torch.transpose(A_, 1, 0)  # K_attention_classesxN_tiles
        A = F.softmax(A_raw, dim=-1)  # normalize attentions scores over tiles
        return A_raw, A

    def forward(self, h):
    
        h = self.compression_layer(h)

        # Attention MIL.
        A_raw, A = self.forward_attention(h) # 1xN tiles
        M = A @ h #torch.Size([1, dim_embedding])  # 1x512 [Sum over N(aihi,1), ..., Sum over N(aihi,512)]

        logits = torch.empty(1, self.n_classes).float().to(h.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat, A_raw, M