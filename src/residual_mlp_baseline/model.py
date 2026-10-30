import torch
import torch.nn as nn

# Color tasks use 64 patches of 4x4 pixels per image; MNIST stays flattened.
# The same model class is used for all three.
TASKS = {
    "mnist":      {"in_features": 28 * 28,     "num_classes": 10},    # 28x28 grayscale
    "cifar10":    {"in_features": 3 * 32 * 32, "num_classes": 10, "num_patches": 64},
    "imagenet32": {"in_features": 3 * 32 * 32, "num_classes": 1000, "num_patches": 64},
}

WIDTH = 128   # Width of the Neural Network
HIDDEN = 256  # Number of neurons in in each hidden layer
DEPTH = 10    # Number of residual blocks in the backbone

class Block(nn.Module):
    def __init__(self, width, hidden, num_patches=None):
        super().__init__()
        if num_patches is not None:
            self.patch_norm = nn.LayerNorm(width)
            self.patch_mlp = nn.Sequential(
                nn.Linear(num_patches, num_patches * 2), nn.SiLU(),
                nn.Linear(num_patches * 2, num_patches)
            )
        self.norm = nn.LayerNorm(width)       # pre-norm: normalize before the layers
        self.fc1 = nn.Linear(width, hidden)   # expand:  width  -> hidden
        self.gate = nn.Linear(width, hidden)  # SwiGLU gating projection
        self.fc2 = nn.Linear(hidden, width)   # project: hidden -> width

    def forward(self, x):
        if hasattr(self, "patch_mlp"):
            x = x + self.patch_mlp(self.patch_norm(x).transpose(1, 2)).transpose(1, 2)
        z = self.norm(x)
        h = self.fc2(nn.functional.silu(self.fc1(z)) * self.gate(z))
        return x + h                          # skip connection: add the input back

class ResidualMLP(nn.Module):
    def __init__(self, in_features, num_classes, width=WIDTH, hidden=HIDDEN, depth=DEPTH,
                 num_patches=None):
        super().__init__()
        self.num_patches = num_patches
        if num_patches is not None:
            grid = int(num_patches ** 0.5)
            if grid < 1 or grid * grid != num_patches or 32 % grid:
                raise ValueError("num_patches must be a positive square whose grid divides 32")
            patch_size = 32 // grid
            self.patchify = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        # Project each patch (or the whole MNIST image) into the backbone.
        self.input_layer = nn.Linear(in_features // (num_patches or 1), width)

        # The residual backbone: `depth` identical blocks.
        self.blocks = nn.ModuleList([Block(width, hidden, num_patches) for _ in range(depth)])

        # The head: same layers as a block, but the last layer outputs class scores.
        self.head_norm = nn.LayerNorm(width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)
        
        

        # Xavier (Glorot) initialization for every linear layer, zero biases.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, streams=None):
        if self.num_patches is not None:
            # [batch, channels, height, width] -> [batch, patches, pixels per patch].
            x = self.patchify(x).transpose(1, 2)
        else:
            x = x.flatten(start_dim=1)
        x = self.input_layer(x)
        if streams is not None:
            x.retain_grad()
            streams.append(x)
        for block in self.blocks:
            x = block(x)
            if streams is not None:
                x.retain_grad()
                streams.append(x)

        if self.num_patches is not None:
            x = x.mean(dim=1)  # pool patches before classification

        # Steps used in each Neuron
        x = self.head_norm(x) # Normilization
        x = self.head_fc1(x) # Linear
        x = nn.functional.silu(x) # Activation
        x = self.head_fc2(x) # Linear

        return x

    def predict(self, x):
        """Inference pass, never used during training. Returns class probabilities."""
        self.eval()                           # switch layers to inference behavior
        with torch.no_grad():                 # no gradients needed at inference
            return torch.softmax(self.forward(x), dim=1)
