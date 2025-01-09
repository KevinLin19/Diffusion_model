import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline

def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ControlNet(nn.Module):
    r"""
    Control Net Module for DDPM
    """
    def __init__(self, model_config,
                 model_locked=True,
                 model_ckpt=None,
                 device=None):
        super().__init__()
        # Trained DDPM
        self.model_locked = model_locked
        #self.trained_unet = Unet(model_config)
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        Unet = pipe.unet
        self.trained_unet = Unet # create the same architecture as the pretrained model

        # Load weights for the trained model
        if model_ckpt is not None and device is not None:
            print('Loading Trained Diffusion Model')
            self.trained_unet.load_state_dict(torch.load(model_ckpt, map_location=device)['model_state_dict'], strict=True)

        # ControlNet Copy of Trained DDPM
        self.control_copy_unet = nn.ModuleDict()  # Use ModuleDict to store named modules
        for name, module in Unet.named_children():
            if name == "up_blocks":
                continue  # Skip up_blocks
            self.control_copy_unet[name] = module  # Add all other components
        # Load same weights as the trained model
        if model_ckpt is not None and device is not None:
            print('Loading Control Diffusion Model')
            self.control_copy_unet.load_state_dict(torch.load(model_ckpt, map_location=device)['model_state_dict'], strict=False)

        # Hint Block for ControlNet
        # Stack of Conv activation and zero convolution at the end
        self.control_copy_unet_hint_block = nn.Sequential(
            nn.Conv2d(model_config['hint_channels'],
                      64,
                      kernel_size=3,
                      padding=(1, 1)),
            nn.SiLU(),
            nn.Conv2d(64,
                      128,
                      kernel_size=3,
                      padding=(1, 1)),
            nn.SiLU(),
            nn.Conv2d(128,
                      self.trained_unet.down_blocks[0].attentions[0].norm.num_groups,
                      kernel_size=3,
                      padding=(1, 1)),
            nn.SiLU(),
            make_zero_module(nn.Conv2d(self.trained_unet.down_blocks[0].attentions[0].norm.num_groups,
                                       self.trained_unet.down_blocks[0].attentions[0].norm.num_groups,
                                       kernel_size=1,
                                       padding=0))
        )

        # Zero Convolution Module for Downblocks(encoder Layers)
        self.control_copy_unet_down_zero_convs = nn.ModuleList([
            make_zero_module(nn.Conv2d(self.trained_unet.down_blocks[i].attentions[0].norm.num_groups,
                                       self.trained_unet.down_blocks[i].attentions[0].norm.num_groups,
                                       kernel_size=1,
                                       padding=0))
            for i in range(len(self.trained_unet.down_blocks)-1)
        ])

        # Zero Convolution Module for MidBlocks
        self.control_copy_unet_mid_zero_convs = nn.ModuleList([
            make_zero_module(nn.Conv2d(self.trained_unet.mid_block.attentions[0].norm.num_groups,
                                       self.trained_unet.mid_block.attentions[0].norm.num_groups,
                                       kernel_size=1,
                                       padding=0))
        ])

    def get_params(self):
        # Add all ControlNet parameters
        # First is our copy of unet
        params = list(self.control_copy_unet.parameters())

        # Add parameters of hint Blocks & Zero convolutions for down/mid blocks
        params += list(self.control_copy_unet_hint_block.parameters())
        params += list(self.control_copy_unet_down_zero_convs.parameters())
        params += list(self.control_copy_unet_mid_zero_convs.parameters())

        # If we desire to not have the decoder layers locked, then add
        # them as well
        if not self.model_locked:
            params += list(self.trained_unet.up_blocks.parameters())
            params += list(self.trained_unet.norm_out.parameters())
            params += list(self.trained_unet.conv_out.parameters())
        return params

    def forward(self, x, t, hint):
        # Time embedding and timestep projection layers of trained unet
        trained_unet_t_emb = get_time_embedding(torch.as_tensor(t).long(),
                                                self.trained_unet.time_embedding.linear_1.in_features)
        trained_unet_t_emb = self.trained_unet.time_embedding(trained_unet_t_emb)

        # Get all downblocks output of trained unet first
        trained_unet_down_outs = []
        with torch.no_grad():
            train_unet_out = self.trained_unet.conv_in(x)
            for idx, down in enumerate(self.trained_unet.down_blocks):
                trained_unet_down_outs.append(train_unet_out)
                train_unet_out = down(train_unet_out, trained_unet_t_emb)

        # ControlNet Layers start here #
        # Time embedding and timestep projection layers of controlnet's copy of unet
        control_copy_unet_t_emb = get_time_embedding(torch.as_tensor(t).long(),
                                                self.control_copy_unet.time_embedding)
        control_copy_unet_t_emb = self.control_copy_unet.time_proj(control_copy_unet_t_emb)

        # Hint block of controlnet's copy of unet
        control_copy_unet_hint_out = self.control_copy_unet_hint_block(hint)

        # Call conv_in layer for controlnet's copy of unet
        # and add hint blocks output to it
        control_copy_unet_out = self.control_copy_unet.conv_in(x)
        control_copy_unet_out += control_copy_unet_hint_out

        # Get all downblocks output for controlnet's copy of unet
        control_copy_unet_down_outs = []
        for idx, down in enumerate(self.control_copy_unet.down_blocks):
            # Save the control nets copy output after passing it through zero conv layers
            control_copy_unet_down_outs.append(
                self.control_copy_unet_down_zero_convs[idx](control_copy_unet_out)
            )
            control_copy_unet_out = down(control_copy_unet_out, control_copy_unet_t_emb)

        
        # Get midblock output of controlnets copy of unet
        control_copy_unet_out = self.control_copy_unet.mid_block(
        control_copy_unet_out,
        control_copy_unet_t_emb
        )

        # Get midblock output of trained unet
        train_unet_out = self.trained_unet.mid_block(train_unet_out, trained_unet_t_emb)

        # Add midblock output of controlnets copy of unet to that of trained unet
        # but after passing them through zero conv layers
        train_unet_out += self.control_copy_unet_mid_zero_convs(control_copy_unet_out)

        # Call upblocks of trained unet
        for up in self.trained_unet.up_blocks:
            # Get downblocks output from both trained unet and controlnets copy of unet
            trained_unet_down_out = trained_unet_down_outs.pop()
            control_copy_unet_down_out = control_copy_unet_down_outs.pop()

            # Add these together and pass this as downblock input to upblock
            train_unet_out = up(train_unet_out,
                                control_copy_unet_down_out + trained_unet_down_out,
                                trained_unet_t_emb)

        # Call output layers of trained unet
        train_unet_out = self.trained_unet.norm_out(train_unet_out)
        train_unet_out = nn.SiLU()(train_unet_out)
        train_unet_out = self.trained_unet.conv_out(train_unet_out)
        # out B x C x H x W
        return train_unet_out






