import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_shape, encoding_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, 6, 2)
        self.conv2 = nn.Conv2d(64, 64, 6, 2, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 6, 2, padding=2)
        self.fc = nn.Linear(self._get_flattened_size(input_shape), encoding_size)

    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def _get_flattened_size(self, input_shape):
        dummy_input = torch.zeros(*input_shape)
        with torch.no_grad():
            x = self._conv_forward(dummy_input)
        return x.numel()

    def forward(self, x: torch.Tensor):
        x = self._conv_forward(x)
        x = x.reshape(x.size(dim=0), -1)
        x = F.relu(self.fc(x))
        return x


class ActCondTrans(nn.Module):
    def __init__(self, encoding_size, feature_size, n_actions):
        super().__init__()
        self.W_h_enc = nn.Linear(encoding_size, feature_size, bias=False)
        self.W_a = nn.Linear(n_actions, feature_size, bias=False)
        self.W_dec = nn.Linear(feature_size, encoding_size)

    def forward(self, h_enc: torch.Tensor, a: torch.Tensor):
        x = self.W_h_enc(h_enc) * self.W_a(a)
        x = self.W_dec(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, n_channels):
        super().__init__()
        self.fc = nn.Linear(input_size, 64 * 10 * 10)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(64, n_channels, kernel_size=6, stride=2)

    def forward(self, h):
        x = F.relu(self.fc(h))
        x = x.view(-1, 64, 10, 10)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x.squeeze(dim=1)


class ObservationPredictionModel(nn.Module):
    """
    Next state predictor as per https://arxiv.org/pdf/1507.08750

    Input shape must be standard Atari framestack: (4, 84, 84)
    """

    def __init__(
        self,
        n_actions,
        encoding_size=1024,
        feature_size=2048,
    ):
        super().__init__()
        input_shape = (4, 84, 84)
        n_channels = 1

        self.encoder = Encoder(input_shape, encoding_size)
        self.action_cond_trans = ActCondTrans(
            encoding_size=encoding_size, feature_size=feature_size, n_actions=n_actions
        )
        self.decoder = Decoder(input_size=encoding_size, n_channels=n_channels)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        h_enc = self.encoder(x)
        h_trans = self.action_cond_trans(h_enc=h_enc, a=a)
        return self.decoder(h_trans)
