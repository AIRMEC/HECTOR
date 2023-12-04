# From EsVIT repo
from esvit.models import build_model
from esvit.config import config, update_config, save_config

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# From EsVIT repo. Need to download full repo. https://github.com/microsoft/esvit
def load_encoder_esVIT(args, device):
    # ============ building network ... ============
    num_features = []
    # if the network is a 4-stage vision transformer (i.e. swin)
    if 'swin' in args.arch :
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        swin_spec = config.MODEL.SPEC
        embed_dim=swin_spec['DIM_EMBED']
        depths=swin_spec['DEPTHS']
        num_heads=swin_spec['NUM_HEADS'] 

        # For each stage, we have n stacked models (d)
        # Each model takes embeddings of dimension embed_dim (the first param), 
        # And then the stage i, input dim is input dim(i-1)*2
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d 

    # if the network is a 4-stage vision transformer (i.e. longformer)
    elif 'vil' in args.arch :
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        msvit_spec = config.MODEL.SPEC
        arch = msvit_spec.MSVIT.ARCH

        layer_cfgs = model.layer_cfgs
        num_stages = len(model.layer_cfgs)
        depths = [cfg['n'] for cfg in model.layer_cfgs]
        dims = [cfg['d'] for cfg in model.layer_cfgs]
        out_planes = model.layer_cfgs[-1]['d']
        Nglos = [cfg['g'] for cfg in model.layer_cfgs]

        print(dims)

        for i, d in enumerate(depths):
            num_features += [ dims[i] ] * d

    # if the network is a 4-stage vision transformer (i.e. CvT)
    elif 'cvt' in args.arch :
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        cvt_spec = config.MODEL.SPEC
        embed_dim=cvt_spec['DIM_EMBED']
        depths=cvt_spec['DEPTH']
        num_heads=cvt_spec['NUM_HEADS'] 


        print(f'embed_dim {embed_dim} depths {depths}')
        
        for i, d in enumerate(depths):
            num_features += [int(embed_dim[i])] * int(d) 

    # if the network is a vanilla vision transformer (i.e. deit_tiny, deit_small, vit_base)
    else:
        raise ValueError(f'{args.arch} not supported yet.')

    model.to(device)

    # load weights to evaluate
    state_dict = torch.load(args.checkpoint, map_location=device)
    # Technically we can also load the weights of the student but in knowledge distillation, I think it's more common to take the teacher 
    # and in DINO paper, they show that the teacher learns better. 
    state_dict = state_dict['teacher']
    #Line below was initally in the code but I think it's usefless in our case (swin-t)
    #state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    #in trained model, you probably have the dense DINO head and in the loaded one a regular head. Those keys won't be matching.
    #IncompatibleKeys(missing_keys=['head.weight', 'head.bias'], unexpected_keys=['head_dense.mlp.0.weight', 'head_dense.mlp.0.bias', 'head_dense.mlp.2.weight', 'head_dense.mlp.2.bias', 'head_dense.mlp.4.weight', 'head_dense.mlp.4.bias', 'head_dense.last_layer.weight_g', 'head_dense.last_layer.weight_v', 'head.mlp.0.weight', 'head.mlp.0.bias', 'head.mlp.2.weight', 'head.mlp.2.bias', 'head.mlp.4.weight', 'head.mlp.4.bias', 'head.last_layer.weight_g', 'head.last_layer.weight_v'])
    #in any case, we do not use the heads but the out features of each stage.
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built with pretrained weigths {args.checkpoint}.")
    
    ##a choice, 4 will take the last 2 stages for instance
    #if n>1, they are just stacked features.
    #paper says : For all transformers architecture, we use the concatenation of view-level features
    # in the last layers (results are similar to the use of 3 or 5 layers in our initial experiments)
    
    num_features_linear = sum(num_features[-args.n_last_blocks:])
    print(f'num_features_linear {num_features_linear}')

    return model, num_features_linear, depths

# Regular resnet encoder. 
def load_encoder_resnet(backbone, checkpoint_file, use_imagenet_weights, device):
    import torch.nn as nn
    import torchvision.models as models

    class DecapitatedResnet(nn.Module):
        def __init__(self, base_encoder, pretrained):
            super(DecapitatedResnet, self).__init__()
            self.encoder = base_encoder(pretrained=pretrained)

        def forward(self, x):
            # Same forward pass function as used in the torchvision 'stock' ResNet code
            # but with the final FC layer removed.
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)

            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)

            x = self.encoder.avgpool(x)
            x = torch.flatten(x, 1)

            return x

    model = DecapitatedResnet(models.__dict__[backbone], use_imagenet_weights)

    if use_imagenet_weights:
        if checkpoint_file is not None:
            raise Exception(
                "Either provide a weights checkpoint or the --imagenet flag, not both."
            )
        print(f"Created encoder with Imagenet weights")
    else:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix from key names
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        # Verify that the checkpoint did not contain data for the final FC layer
        msg = model.encoder.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print(f"Loaded checkpoint {checkpoint_file}")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    return model

