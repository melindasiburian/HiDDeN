import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    Now supports mask-aware embedding with optional mask and alpha parameters.

    Input: cover image and watermark message (and optionally mask and alpha)
    Module inserts watermark into image (encoded_image), then applies Noise layers
    (noised_image), then passes noised_image to Decoder which tries to recover
    the watermark (decoded_message).

    Output: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser
        self.decoder = Decoder(config)

    def forward(self, image, message, mask=None, alpha=None):
        """
        Forward pass with optional mask and alpha.

        Args:
            image: Cover image tensor [B, 3, H, W], range [-1, 1]
            message: Message tensor [B, message_length]
            mask: Optional binary mask tensor.
                  If None, uses all-one mask (embed everywhere).
            alpha: Optional embedding strength.
                   If None, uses default value (1.0).

        Returns:
            encoded_image: Encoded stego image [B, 3, H, W]
            noised_image: Noised stego image [B, 3, H, W]
            decoded_message: Decoded message [B, message_length]
        """
        # Pass mask and alpha to encoder
        encoded_image = self.encoder(image, message, mask=mask, alpha=alpha)

        # Pass encoded image to noiser
        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]

        # Pass noised image to decoder
        decoded_message = self.decoder(noised_image)

        return encoded_image, noised_image, decoded_message


class EncoderDecoderOriginal(nn.Module):
    """
    Original EncoderDecoder for backward compatibility.
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):
        super(EncoderDecoderOriginal, self).__init__()
        from model.encoder import EncoderOriginal
        self.encoder = EncoderOriginal(config)
        self.noiser = noiser
        self.decoder = Decoder(config)

    def forward(self, image, message):
        """
        Original forward without mask and alpha.
        """
        encoded_image = self.encoder(image, message)
        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message