import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AttnNet(nn.Module):
    # Adapted from https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
    # Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w

    def __init__(self, L=1024, D=256, dropout=False, p_dropout_atn=0.25, n_classes=1):
        super(AttnNet, self).__init__()

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

class Attn_Modality_Gated(nn.Module):
    # Adapted from https://github.com/mahmoodlab/PORPOISE
    def __init__(self, gate_h1, gate_h2, gate_h3, dim1_og, dim2_og, dim3_og, use_bilinear=[True,True,True], scale=[1,1,1], p_dropout_fc=0.25):
        super(Attn_Modality_Gated, self).__init__()

        self.gate_h1 = gate_h1 #[boolean]
        self.gate_h2 = gate_h2 #[boolean]
        self.gate_h3 = gate_h3 #[boolean]
        self.use_bilinear = use_bilinear #[boolean]

        # can perform attention on latent vectors of lower dimension
        dim1, dim2, dim3 = dim1_og//scale[0], dim2_og//scale[1], dim3_og//scale[2]

        # attention gate of each modality
        if self.gate_h1:
            self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
            self.linear_z1 = nn.Bilinear(dim1_og, dim2_og+dim3_og, dim1) if self.use_bilinear[0] else nn.Sequential(nn.Linear(dim1_og+dim2_og+dim3_og, dim1))
            self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h1, self.linear_o1 = nn.Identity(), nn.Identity()  

        if self.gate_h2:
            self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
            self.linear_z2 = nn.Bilinear(dim2_og, dim1_og+dim3_og, dim2) if self.use_bilinear[1] else nn.Sequential(nn.Linear(dim1_og+dim2_og+dim3_og, dim2))
            self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h2, self.linear_o2 = nn.Identity(), nn.Identity()  

        if self.gate_h3:
            self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
            self.linear_z3 = nn.Bilinear(dim3_og, dim1_og+dim2_og, dim3) if self.use_bilinear[2] else nn.Sequential(nn.Linear(dim1_og+dim2_og+dim3_og, dim3))
            self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h3, self.linear_o3 = nn.Identity(), nn.Identity()  

    def forward(self, x1, x2, x3):

        if self.gate_h1:
            h1 = self.linear_h1(x1) #breaks colli of h1
            z1 = self.linear_z1(x1, torch.cat([x2,x3], dim=-1)) if self.use_bilinear[0] else self.linear_z1(torch.cat((x1, x2, x3), dim=-1)) #creates a vector combining both modalities
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1) #update modality input
        else:
            h1 = self.linear_h1(x1)
            o1 = self.linear_o1(h1)

        if self.gate_h2:
            h2 = self.linear_h2(x2)
            z2 = self.linear_z2(x2, torch.cat([x1,x3], dim=-1)) if self.use_bilinear[1] else self.linear_z2(torch.cat((x1, x2, x3), dim=-1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(x2)
            o2 = self.linear_o2(h2)
        
        if self.gate_h3:
            h3 = self.linear_h3(x3)
            z3 = self.linear_z3(x3, torch.cat([x1,x2], dim=-1)) if self.use_bilinear[2] else self.linear_z3(torch.cat((x1, x2, x3), dim=-1))
            o3 = self.linear_o3(nn.Sigmoid()(z3)*h3)
        else:
            h3 = self.linear_h3(x3)
            o3 = self.linear_o3(h3)

        return o1, o2, o3

class FC_block(nn.Module):
    def __init__(self, dim_in, dim_out, act_layer=nn.ReLU, dropout=True, p_dropout_fc=0.25):
        super(FC_block, self).__init__()

        self.fc = nn.Linear(dim_in, dim_out)
        self.act = act_layer()
        self.drop = nn.Dropout(p_dropout_fc) if dropout else nn.Identity()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class Categorical_encoding(nn.Module):
    def __init__(self, taxonomy_in=3, embedding_dim=128, depth=1, act_fct='relu', dropout=True, p_dropout=0.25):
        super(Categorical_encoding, self).__init__()

        act_fcts = {'relu': nn.ReLU(),
        'elu' : nn.ELU(),
        'tanh': nn.Tanh(),
        'selu': nn.SELU(),
        }
        dropout_module = nn.AlphaDropout(p_dropout) if act_fct=='selu' else nn.Dropout(p_dropout)

        self.embedding = nn.Embedding(taxonomy_in, embedding_dim)
        
        fc_layers = []
        for d in range(depth):
            fc_layers.append(nn.Linear(embedding_dim//(2**d), embedding_dim//(2**(d+1))))
            fc_layers.append(dropout_module if dropout else nn.Identity())
            fc_layers.append(act_fcts[act_fct])

        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.fc_layers(x)
        return x

class HECTOR(nn.Module):
    def __init__(
        self,
        input_feature_size=1024,
        precompression_layer=True,
        feature_size_comp = 512,
        feature_size_attn = 256,
        postcompression_layer=True,
        feature_size_comp_post = 128,
        dropout=True,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
        n_classes=2,
        input_stage_size=6,
        embedding_dim_stage=16,
        depth_dim_stage=1,
        act_fct_stage='elu',
        dropout_stage=True,
        p_dropout_stage=0.25,
        input_mol_size=4,
        embedding_dim_mol=16,
        depth_dim_mol=1,
        act_fct_mol='elu',
        dropout_mol=True,
        p_dropout_mol=0.25,
        fusion_type='kron',
        use_bilinear=[True,True,True],
        gate_hist=False,
        gate_stage=False,
        gate_mol=False,
        scale=[1,1,1],
    ):
        super(HECTOR, self).__init__()

        self.fusion_type =fusion_type
        self.input_stage_size=input_stage_size
        self.use_bilinear = use_bilinear
        self.gate_hist = gate_hist
        self.gate_stage = gate_stage
        self.gate_mol = gate_mol

        # Reduce dimension of H&E patch features.
        if precompression_layer:
            self.compression_layer = nn.Sequential(*[FC_block(input_feature_size, feature_size_comp*4, p_dropout_fc=p_dropout_fc),
                                                    FC_block(feature_size_comp*4, feature_size_comp*2, p_dropout_fc=p_dropout_fc),
                                                    FC_block(feature_size_comp*2, feature_size_comp, p_dropout_fc=p_dropout_fc),])

            dim_post_compression = feature_size_comp
        else:
            self.compression_layer = nn.Identity()
            dim_post_compression = input_feature_size

        # Get embeddings of categorical features.
        self.encoding_stage_net = Categorical_encoding(taxonomy_in=self.input_stage_size, 
                                                embedding_dim=embedding_dim_stage, 
                                                depth=depth_dim_stage, 
                                                act_fct=act_fct_stage, 
                                                dropout=dropout_stage, 
                                                p_dropout=p_dropout_stage)
        self.out_stage_size = embedding_dim_stage//(2**depth_dim_stage)
     
        self.encoding_mol_net = Categorical_encoding(taxonomy_in=input_mol_size, 
                                                embedding_dim=embedding_dim_mol, 
                                                depth=depth_dim_mol, 
                                                act_fct=act_fct_mol, 
                                                dropout=dropout_mol, 
                                                p_dropout=p_dropout_mol)
        h_mol_size_out = embedding_dim_mol//(2**depth_dim_mol)

        # For survival tasks the attention scores are binary (set to class=1).
        self.attention_survival_net = AttnNet(
            L=dim_post_compression,
            D=feature_size_attn,
            dropout=dropout,
            p_dropout_atn=p_dropout_atn,
            n_classes=1,)

        # Attention gate on each modality.
        self.attn_modalities = Attn_Modality_Gated(
            gate_h1=self.gate_hist, 
            gate_h2=self.gate_stage, 
            gate_h3=self.gate_mol,
            dim1_og=dim_post_compression, 
            dim2_og=self.out_stage_size, 
            dim3_og=h_mol_size_out,
            scale=scale, 
            use_bilinear=self.use_bilinear)

        # Post-compression layer for H&E slide-level embedding before fusion.
        dim_post_compression = dim_post_compression//scale[0] if self.gate_hist else dim_post_compression
        self.post_compression_layer_he = FC_block(dim_post_compression, dim_post_compression//2, p_dropout_fc=p_dropout_fc)
        dim_post_compression = dim_post_compression//2

        # Post-compression layer.
        dim1, dim2, dim3 = dim_post_compression, self.out_stage_size//scale[1] if self.gate_stage else self.out_stage_size, h_mol_size_out//scale[2] if self.gate_mol else h_mol_size_out
        if self.fusion_type=='bilinear':
            head_size_in = (dim1+1)*(dim2+1)*(dim3+1)
        elif self.fusion_type=='kron':
            head_size_in = (dim1)*(dim2)*(dim3)
        elif self.fusion_type=='concat':
            head_size_in = dim1+dim2+dim3

        self.post_compression_layer = nn.Sequential(*[FC_block(head_size_in, feature_size_comp_post*2, p_dropout_fc=p_dropout_fc),
                                                        FC_block(feature_size_comp_post*2, feature_size_comp_post, p_dropout_fc=p_dropout_fc),])

        # Survival head.
        self.n_classes = n_classes
        self.classifier = nn.Linear(feature_size_comp_post, self.n_classes)

        # Init weights.
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward_attention(self, h):
        A_ = self.attention_survival_net(h)  # h shape is N_tilesxdim
        A_raw = torch.transpose(A_, 1, 0)  # K_attention_classesxN_tiles
        A = F.softmax(A_raw, dim=-1)  # #normalize attentions scores over tiles
        return A_raw, A

    def forward_fusion(self, h1, h2, h3):

        if self.fusion_type=='bilinear':
            # Append 1 to retain unimodal embeddings in the fusion
            h1 = torch.cat((h1, torch.ones(1, 1, dtype=torch.float, device=h1.device)), -1)
            h2 = torch.cat((h2, torch.ones(1, 1, dtype=torch.float, device=h2.device)), -1)
            h3 = torch.cat((h3, torch.ones(1, 1, dtype=torch.float, device=h3.device)), -1)

            return torch.kron(torch.kron(h1, h2), h3)

        elif self.fusion_type=='kron':
            return torch.kron(torch.kron(h1, h2), h3)

        elif self.fusion_type=='concat':
            return torch.cat([h1, h2, h3], dim=-1)
        else:
            print('Not implemeted') 
            #raise Exception ... 

    def forward_survival(self, logits):
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Model outputs the hazards with sigmoid activation function.
        hazards = torch.sigmoid(logits) #size [1, n_classes] h(t|X) := P(T=t|T>=t,X)
        #S(t|X) := P(T>=t|X) = TT (1-h(s|X)) for s=1,t. This is computed for each discrete time point t. So for s=1 there is no cum prod. 
        survival = torch.cumprod(1 - hazards, dim=1) #size [1, n_classes]

        return hazards, survival, Y_hat

    def forward(self, h, stage, h_mol):

        # H&E embedding.
        h = self.compression_layer(h)

        # Attention MIL and first-order pooling.
        A_raw, A = self.forward_attention(h) # 1xN tiles
        h_hist = A @ h #torch.Size([1, dim_embedding]) [Sum over N(aihi,1), ..., Sum over N(aihi,dim_embedding)]
        
        # Stage learnable embedding.
        stage = self.encoding_stage_net(stage)

        # Compression h_mol.
        h_mol = self.encoding_mol_net(h_mol)

        # Attention gates on each modality.
        h_hist, stage, h_mol = self.attn_modalities(h_hist, stage, h_mol)

        # Post-compressiong H&E slide embedding.
        h_hist = self.post_compression_layer_he(h_hist)

        # Fusion.
        m = self.forward_fusion(h_hist, stage, h_mol)

        # Post-compression of multimodal embedding.
        m = self.post_compression_layer(m)

        # Survival head.
        logits  = self.classifier(m)

        hazards, survival, Y_hat = self.forward_survival(logits)

        return hazards, survival, Y_hat, A_raw, m
