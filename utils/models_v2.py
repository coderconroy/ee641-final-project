import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

class EMA(nn.Module):
    """
    Manages Exponential Moving Averages for neural network parameters in PyTorch.
    """
    def __init__(self, decay_rate):
        super(EMA, self).__init__()
        self.decay_rate = decay_rate
        self.step_count = 0

    def apply_parameter_smoothing(self, target_model, source_model):
        """ Smoothens the parameters of the target model based on the source model. """
        for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
            target_param.data = self.compute_moving_average(target_param.data, source_param.data)

    def compute_moving_average(self, target, source):
        """ Calculates the moving average of the parameters. """
        if target is None:
            return source
        return target * self.decay_rate + (1 - self.decay_rate) * source

    def step_ema(self, target_model, source_model, init_step=2000):
        """ Updates the EMA model's parameters or initializes them depending on the iteration count. """
        if self.step_count < init_step:
            self.init_params(target_model, source_model)
        else:
            self.apply_parameter_smoothing(target_model, source_model)
        self.step_count += 1

    def init_params(self, target_model, source_model):
        """ Initializes the target model's parameters with those from the source model. """
        target_model.load_state_dict(source_model.state_dict())

class AttentionMechanism(nn.Module):
    """
    Self-Attention Mechanism with Multi-Head Attention followed by a feed-forward network.
    Processes input tensor with spatial dimensions reshaped as a sequence.
    """
    def __init__(self, num_channels, spatial_size):
        super(AttentionMechanism, self).__init__()
        self.num_channels = num_channels
        self.spatial_size = spatial_size

        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(num_channels, 4, batch_first=True)

        # Layer normalization
        self.norm_layer = nn.LayerNorm([num_channels])

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm([num_channels]),
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels)
        )

    def forward(self, x):
        # Reshape and transpose the input tensor for multi-head attention
        x = x.view(-1, self.num_channels, self.spatial_size ** 2).transpose(1, 2)
        x_norm = self.norm_layer(x)

        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(x_norm, x_norm, x_norm)

        # Combine the attention output with the original tensor and apply the feed-forward network
        att_output = attn_output + x
        att_output = self.ffn(att_output) + att_output

        # Reshape the output back to the original spatial dimensions
        return att_output.transpose(2, 1).view(-1, self.num_channels, self.spatial_size, self.spatial_size)
    
class ConvBlock(nn.Module):
    """
    Convolutional Block consisting of two convolutional layers, each followed by Group Normalization and GELU activation applied to the first layer.
    Optionally includes a residual connection.
    """
    def __init__(self, c_in, c_out, c_hidden=None, use_residual=False, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual

        # Initialize hidden layer channels to c_hidden if defined, else c_out
        c_hidden = c_hidden or c_out

        # Convolutional layers with Group Normalization and GELU activation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(1, c_hidden),
            nn.GELU(),
            nn.Conv2d(c_hidden, c_out, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(1, c_out)
        )

    def forward(self, x):
        # Apply convolutional layers and optionally residual connection
        x_out = self.conv_layers(x)
        return F.gelu(x_out + x) if self.use_residual else x_out

class DownscaleBlock(nn.Module):
    """
    Downscaling block with max pooling followed by two convolutional blocks.
    Integrates a time-dependent embedding into the feature map.
    """
    def __init__(self, c_in, c_out, embed_size=256):
        super(DownscaleBlock, self).__init__()

        # Pooling and convolutional operations
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            ConvBlock(c_in, c_in, use_residual=True),
            ConvBlock(c_in, c_out)
        )

        # Embedding layer for time and/or class dependent features
        self.embed_layer = nn.Linear(embed_size, c_out)

    def forward(self, x, embed):
        # Apply pooling, convolution and embedding layers
        x = self.pool(x)
        x = self.conv(x)

        # Apply embedding layers and duplicate the embedding to match the feature spatial dimensions
        embed = F.silu(embed)
        embed = self.embed_layer(embed)
        embed = embed.reshape(-1, embed.shape[1], 1, 1)
        embed = embed.expand(-1, -1, x.shape[-2], x.shape[-1])

        # Combine the features with the embedding
        return x + embed


class UpscaleBlock(nn.Module):
    """
    Upscaling block with bilinear upsampling followed by two convolutional blocks.
    Integrates a time-dependent embedding into the feature map and combines it with skip connections.
    """
    def __init__(self, c_in, c_out, embed_size=256):
        super(UpscaleBlock, self).__init__()

        # Upsampling and convolutional operations
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_layers = nn.Sequential(
            ConvBlock(c_in, c_in, use_residual=True),
            ConvBlock(c_in, c_out, kernel_size=3)
        )

        # Embedding layer for time and/or class dependent features
        self.embed_layer = nn.Linear(embed_size, c_out)

    def forward(self, x, x_skip, embed):
        # Apply upsampling
        x = self.upsample(x)

        # Concatenate the with skip connection features and apply convolution
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv_layers(x)

        # Apply embedding layers and duplicate the embedding to match the feature spatial dimensions
        embed = F.silu(embed)
        embed = self.embed_layer(embed)
        embed = embed.reshape(-1, embed.shape[1], 1, 1)
        embed = embed.expand(-1, -1, x.shape[-2], x.shape[-1])

        # Combine the features with the embedding
        return x + embed

class Encoder(nn.Module):
    def __init__(self, c_in):
        super(Encoder, self).__init__()
        # Initial convolution block to process the input image
        self.initial_conv = ConvBlock(c_in, 64)

        # Downsampling path with ConvBlocks and self-attention mechanisms
        self.down1 = DownscaleBlock(64, 128)
        self.attention1 = AttentionMechanism(128, 32)
        self.down2 = DownscaleBlock(128, 256)
        self.attention2 = AttentionMechanism(256, 16)
        self.down3 = DownscaleBlock(256, 256)
        self.attention3 = AttentionMechanism(256, 8)

    def forward(self, x, embed):
        x0 = self.initial_conv(x)
        x1 = self.down1(x0, embed)
        x1 = self.attention1(x1)
        x2 = self.down2(x1, embed)
        x2 = self.attention2(x2)
        x3 = self.down3(x2, embed)
        x3 = self.attention3(x3)
        return [x0, x1, x2, x3]
    
class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck, self).__init__()
        self.bottleneck1 = ConvBlock(256, 512)
        self.bottleneck2 = ConvBlock(512, 512)
        self.bottleneck3 = ConvBlock(512, 256)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, c_out):
        super(Decoder, self).__init__()
        # Upsampling path with ConvBlocks, self-attention mechanisms, and skip connections
        self.up1 = UpscaleBlock(512, 128)
        self.attention4 = AttentionMechanism(128, 16)
        self.up2 = UpscaleBlock(256, 64)
        self.attention5 = AttentionMechanism(64, 32)
        self.up3 = UpscaleBlock(128, 64)
        self.attention6 = AttentionMechanism(64, 64)

        # Final convolutional layer to produce the output image or feature map
        self.output_conv = nn.Conv2d(64, c_out, kernel_size=1)

    def forward(self, x_layers, embed):
        # Decoding (upsampling) path with skip connections
        x = self.up1(x_layers[3], x_layers[2], embed)
        x = self.attention4(x)
        x = self.up2(x, x_layers[1], embed)
        x = self.attention5(x)
        x = self.up3(x, x_layers[0], embed)
        x = self.attention6(x)

        # Final output layer
        x = self.output_conv(x)
        return x
    

class UNet(nn.Module):
    """
    UNet architecture with added self-attention mechanism and time-dependent embeddings.
    This architecture is typically used for tasks like image segmentation and conditional image generation.
    """
    def __init__(self, device, c_in=3, c_out=3, embed_size=256, num_classes=None):
        super(UNet, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.num_classes = num_classes

        # Initialize encoder, bottleneck and decoder
        self.encoder = Encoder(c_in)
        self.bottleneck = BottleNeck()
        self.decoder = Decoder(c_out)

        # Initialize label embedding layer if conditional diffusion is being performed
        if num_classes is not None:
            self.y_embed_layer = nn.Embedding(num_classes, embed_size)

    def load_state_dict(self, state_dict, strict=True):
        """
        Overrides the load_state_dict method to update y_embed_layer based on num_classes.
        """
        # Create embed layer since it is not created in the constructor since the embed size is unknown until the state
        # is loaded
        if 'y_embed_layer.weight' in state_dict:
            embed_shape = state_dict['y_embed_layer.weight'].cpu().numpy().shape
            self.num_classes = embed_shape[0]
            self.embed_size = embed_shape[1]
            self.y_embed_layer = nn.Embedding(self.num_classes, self.embed_size).to(self.device)

        # Call the original load_state_dict
        super(UNet, self).load_state_dict(state_dict, strict)
            
    def embed_time(self, t, channels):
        # Generates sinusoidal positional encodings.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        sin_term = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        cos_term = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([sin_term, cos_term], dim=-1)
        return pos_enc

    def forward(self, x, t, y=None):
        # Create embedding of time step information and label information (if conditional diffusion)
        t = t[:, None].type(torch.float)
        t_embed = self.embed_time(t, self.embed_size)
        embed = t_embed if y is None else t_embed + self.y_embed_layer(y)

        # Pass image and embedding through UNet
        x_layers = self.encoder(x, embed)
        x_layers[-1] = self.bottleneck(x_layers[-1])
        out = self.decoder(x_layers, embed)
        
        return out