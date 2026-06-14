import torch
import torch.nn as nn
import torch.nn.functional as F
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    Now supports mask-aware embedding: encoded_image = image + alpha * mask * residual
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks
        self.message_length = config.message_length

        # Store default alpha for backward compatibility
        self.default_alpha = 1.0

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)

        # After concat layer: takes (message + encoded_features + image + mask) channels
        # message_length + conv_channels + 3 + 1 (for mask)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length + 1,
                                             self.conv_channels)

        # Final layer outputs 3-channel residual instead of direct encoded image
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message, mask=None, alpha=None):
        """
        Forward pass with optional mask and alpha for mask-aware embedding.

        Args:
            image: Cover image tensor [B, 3, H, W], range [-1, 1]
            message: Message tensor [B, message_length]
            mask: Binary mask tensor [B, 1, H, W] or [B, H, W] or [1, H, W].
                  If None, uses all-one mask (embed everywhere).
            alpha: Embedding strength. If None, uses default_alpha (1.0).

        Returns:
            encoded_image: Encoded stego image tensor [B, 3, H, W], range [-1, 1]
        """
        # Handle default mask (all ones)
        if mask is None:
            mask = torch.ones_like(image[:, :1, :, :])
        else:
            # Ensure mask has correct shape [B, 1, H, W]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [H, W] -> [1, H, W]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
            # Broadcast to batch size
            if mask.shape[0] != image.shape[0]:
                mask = mask.expand(image.shape[0], 1, -1, -1)

        # Handle default alpha
        if alpha is None:
            alpha = self.default_alpha

        # Ensure alpha is tensor with correct shape for broadcasting
        if isinstance(alpha, (int, float)):
            alpha = torch.tensor(alpha, dtype=image.dtype, device=image.device)
        if alpha.dim() == 0:
            alpha = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # scalar -> [1, 1, 1, 1]
        elif alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)  # [B] -> [B, 1, 1, 1]

        # Expand message to image dimensions
        expanded_message = message.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        expanded_message = expanded_message.expand(-1, -1, self.H, self.W)  # [B, L, H, W]

        # Get CNN features
        encoded_image = self.conv_layers(image)

        # Concatenate: expanded_message + encoded_features + image + mask
        concat = torch.cat([expanded_message, encoded_image, image, mask], dim=1)

        # Process through after_concat_layer
        im_w = self.after_concat_layer(concat)

        # Get residual from final layer
        residual = self.final_layer(im_w)  # [B, 3, H, W]

        # Apply mask-aware embedding: encoded = image + alpha * mask * residual
        # Ensure mask and alpha have correct dimensions for broadcasting
        mask_for_mult = mask.expand(-1, 3, -1, -1)  # [B, 1, H, W] -> [B, 3, H, W]
        encoded_image = image + alpha * mask_for_mult * residual

        # Clamp to valid image range [-1, 1]
        encoded_image = torch.clamp(encoded_image, -1, 1)

        return encoded_image


class EncoderOriginal(nn.Module):
    """
    Original encoder for backward compatibility.
    This is the version before mask-aware embedding was added.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(EncoderOriginal, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):
        """
        Original forward without mask and alpha.
        """
        expanded_message = message.unsqueeze(-1).unsqueeze(-1)
        expanded_message = expanded_message.expand(-1, -1, self.H, self.W)

        encoded_image = self.conv_layers(image)
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w