
from torch import nn


class MainGan(nn.Module):
    def __init__(self, data_depth, coder, critic, device):
        super(MainGan, self).__init__()
        self.encoder = coder(data_depth, True)
        self.decoder = coder(data_depth, False)
        self.critic = critic()
        self.to_device(device)

    def to_device(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.critic.to(device)

    def critic_params(self):
        return self.critic.parameters()

    def coder_params(self):
        return list(self.decoder.parameters()) + list(self.encoder.parameters())