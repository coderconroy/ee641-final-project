import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import torch

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

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

        # Initialize encoder, bottleneck and decoder
        self.encoder = Encoder(c_in)
        self.bottleneck = BottleNeck()
        self.decoder = Decoder(c_out)

        # Initialize label embedding layer if conditional diffusion is being performed
        if num_classes is not None:
            self.y_embed_layer = nn.Embedding(num_classes, embed_size)

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