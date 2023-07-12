import torch
from torch import nn

from transformers.models.vit.configuration_vit import ViTConfig

class GlobalDecoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super(GlobalDecoder, self).__init__()
        
        patch_size, hidden_size, image_size = config.patch_size, config.hidden_size, config.image_size
        sq_num_patch = int(image_size/patch_size)
        num_patch = int(sq_num_patch**2)

        self.model = nn.Sequential()
        dims = [hidden_size, hidden_size//4, hidden_size//16, hidden_size//32, hidden_size//64]
        sizes = [sq_num_patch, 4, 2, 2, 2]

        for idx in range(len(dims)-1):
            input_dim, output_dim = dims[idx], dims[1+idx]
            size = sizes[idx]

            layer = nn.Sequential(
                nn.ConvTranspose2d(input_dim, output_dim, (size, size), stride=size),
                nn.BatchNorm2d(output_dim),
                nn.GELU(),
                nn.Dropout2d(p=0.3),
            )
            self.model.add_module(f"convTranspose{1+idx}", layer)

        input_dim, output_dim = dims[-1], 1
        size = sizes[-1]
        layer = nn.Sequential(
                nn.ConvTranspose2d(input_dim, output_dim, (size, size), stride=size),
                nn.Tanh()
            )
        self.model.add_module(f"convTranspose{2+idx}", layer)

    def forward(self, x):
        return self.model(x)