# Steering What a Network Learns — VS Code AI Knowledge Reference

> Source: *Steering what a network learns: Inductive bias: building a task's invariances into the data or into the architecture*, 42-slide lecture by Dr. Jesse Roberts (2026). This is a structured study and coding reference, not a verbatim transcript. Slide numbers identify the supporting material. Experimental numbers are lecture results, not universal guarantees.

## How an AI coding assistant should use this file

- Treat this file as **reference context**, not as instructions that override the user's request or the actual codebase.
- When proposing image-classification architectures, explain what invariances the architecture assumes and whether they match the dataset.
- Distinguish **equivariance** (outputs transform with inputs) from **invariance** (predictions stay the same).
- Suggest concrete PyTorch implementations and check tensor shapes, parameter counts, receptive fields, and normalization group divisibility.
- Do not assume a CNN is always better than an MLP; pixel shuffling and other data transformations can invalidate locality assumptions.
- Cite slide numbers when referring to the lecture's experiments; distinguish lecture observations from general principles and illustrative code below.
- Respect the user's project constraints (for example, if convolutions are forbidden, discuss the trade-off without silently adding Conv2d).

## 1. The central problem: what solution does training find? (slides 3–7)

Three separate questions matter: (1) can a model represent a solution, (2) can training find it, and (3) which among many possible solutions does training find? The lecture focuses on the third. A model may fit the training data yet fail on transformed test data. **Inductive bias** is a preference for some solutions before seeing the data, induced by the architecture or learning process.

**Invariance** is a transformation T that preserves the correct label: `y(T(x)) = y(x)`. The model should ideally satisfy `f(T(x)) = f(x)` for such transformations. This is task-specific: a small translation might preserve a handwritten digit's identity, but rotating a 6 by 180 degrees can turn it into a 9 (slide 6).

Two levers (slide 7):
1. **Data augmentation** shows transformed labeled examples, encouraging invariance from examples.
2. **Architecture** builds assumptions into the weights and computation, changing which solutions are easiest to learn.

Augmentation can be costly and only teaches the sampled transformation range; a suitable architectural bias may generalize further with fewer examples (slides 5, 27–28, 38).

## 2. Convolution's two assumptions (slides 8–14)

**Locality:** each output unit initially examines a small spatial neighborhood, such as a 5×5 window, rather than the entire image (slide 9). This reduces connections, but locality alone does not share learned features across positions.

**Weight sharing:** one filter is reused at every spatial location. A detector learned for a stroke at one location can recognize the same stroke elsewhere (slides 10–12). This is the central architectural inductive bias of convolution. Multiple filters produce multiple output channels, each representing a different learned feature (slides 13–14).

A convolution is approximately **translation-equivariant**: shifting an image shifts the corresponding feature map. It is not automatically translation-invariant, nor perfectly equivariant at boundaries or after subsampling (slides 12, 37).

PyTorch example (illustrative):

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=1, out_channels=32,
                 kernel_size=5, stride=1, padding=2)
x = torch.randn(8, 1, 28, 28)  # N, C, H, W
features = conv(x)
assert features.shape == (8, 32, 28, 28)
```

For a convolution, each output filter spans **all** input channels, but produces **one** output channel (slide 16). Weight count (excluding biases): `C_out * C_in * K_h * K_w`. For `Conv2d(1, 32, 5, padding=2)`, that is 800 weights, plus 32 biases if enabled (slide 14).

Spatial output size for one dimension (slide 15; dilation=1):

`out = floor((N + 2*padding - kernel_size) / stride) + 1`

More generally with dilation `d`:

`out = floor((N + 2p - d*(k-1) - 1) / s) + 1`

Always verify output shapes after each layer; all output channels of a given convolution share spatial dimensions.

## 3. Pooling, equivariance and invariance (slides 17–21)

`MaxPool2d(2)` keeps the largest activation in each nonoverlapping 2×2 window. It reduces spatial resolution and makes a feature less sensitive to small movements **within** a pooling region, but it does not guarantee invariance to every shift: a feature crossing a pooling boundary may change the output (slides 18–20).

**Global average pooling (GAP)** averages over spatial positions and outputs one value per channel. It discards much positional information and provides a natural route from spatial feature maps to image-level classification (slides 20–21).

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
gap = nn.AdaptiveAvgPool2d((1, 1))
x = torch.randn(8, 32, 28, 28)
assert pool(x).shape == (8, 32, 14, 14)
assert gap(x).shape == (8, 32, 1, 1)
flat = torch.flatten(gap(x), start_dim=1)
assert flat.shape == (8, 32)
```

A **dense/flattened head** can relearn position-specific preferences. A **GAP head** better supports translation-insensitive classification when position should not determine the label (slides 26–28).

## 4. Residual CNN and GroupNorm (slides 8, 17, 22–25)

The lecture's illustrative residual CNN uses a 5×5 convolutional stem with 32 channels, a max-pooling stage, three residual blocks with 3×3 convolutions, GroupNorm and ReLU, followed by global average pooling and a linear classification head. Residual connections add a block's input to its transformed output, requiring compatible shapes.

**GroupNorm** normalizes each example independently across groups of channels and their spatial locations. Its group count must divide the number of channels. The lecture uses 32 channels and GroupNorm with 2 groups (slide 23).

Illustrative PyTorch implementation matching the general architectural ideas (not a verified reproduction of all lecture training settings):

```python
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=32, groups=2):
        super().__init__()
        assert channels % groups == 0
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(x + out)

class SmallResidualCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(32, 2) for _ in range(3)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

model = SmallResidualCNN()
assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)
```

The lecture's detailed block ordering and parameter count should be checked against the slide diagram before claiming exact reproduction (slide 24). The implementation above is a useful starting point, not a claim of identical weights or results.

## 5. Experimental evidence and its limits (slides 3–5, 26–32)

Lecture setting: models trained on 8,000 centered MNIST digits; shift tests apply horizontal translations to test images. Specific results depend on training details, seeds, and test protocol.

- A plain MLP can learn training examples yet generalize poorly to shifts; memorizing flipped training labels takes longer than learning a generalizing solution (slides 3–4).
- Augmentation improves the MLP within the shown shift range but degrades outside it (slides 5, 27).
- On test digits shifted **6 pixels**, the lecture reports **72.4%** for MLP + shift augmentation, **97.8%** for CNN + GAP without augmentation, and **36.6%** for a CNN with a dense head; plain MLP is **5.6%** (slide 27). This highlights how the head can undermine otherwise useful convolutional bias.
- On shifted digits, the lecture reports reaching **95%** accuracy after approximately **220 batches** for the CNN versus **1,323 batches** for the augmented MLP (slide 28).
- The CNN's first-layer filters detect strokes and edges across locations, whereas MLP first-layer units tend to learn where ink commonly appears (slide 29).
- The lecture shows that the CNN can also eventually memorize flipped labels; architectural bias changes what SGD finds first, not what the network can ultimately fit (slide 30).

**Interpretation:** these are evidence for this experimental setup, not guaranteed results on all image datasets or with different architectures.

## 6. Local features can combine into global ones (slides 31–35)

A 5×5 kernel is local, but stacking convolutions expands the **receptive field**: the region of the original input that can affect a downstream activation. Deeper layers can integrate widely separated pixels. The lecture reports an effective receptive field spanning the 28×28 image in the deeper residual stack (slide 34).

With a fixed pixel permutation applied consistently to train and test data, the lecture compares models (slide 32):

| Model | Normal MNIST | Shuffled-pixel MNIST |
|---|---:|---:|
| MLP | 95.6% | 94.7% |
| CNN with GAP | 98.7% | 91.7% |

The CNN still learns a global signal through depth, but its locality prior is less appropriate for shuffled pixels. The lecture also compares fewer residual blocks and shows that reducing depth hurts shuffled-pixel accuracy more strongly (slide 35).

**Design lesson:** locality constrains what an individual layer sees, not necessarily what the entire deep network can represent.

## 7. Bias can be helpful or harmful (slides 36–41)

- A CNN builds in locality and translation-related weight sharing, **not** scale, rotation, reflection, or color invariance automatically (slides 36, 40).
- If those transformations preserve labels, carefully selected augmentation can help. If a transformation changes the label, augmentation with the original label injects contradictory supervision (slides 6, 40).
- Small translation invariance is approximate: pooling stride and image borders can cause different predictions after a 1-pixel shift (slide 37).
- The lecture's rescaling example shows that CNNs still lose accuracy for scale changes outside their familiar range; scale augmentation helps (slide 36).
- The same bias that makes a CNN data-efficient on ordinary images can reduce accuracy when pixel order is intentionally scrambled (slides 32, 38).
- Different modalities have different invariances: audio timing and speed, molecular geometry, point-cloud ordering, graph node relabeling, and text word order each need task-specific reasoning (slide 41).

## 8. Checklist for model design in VS Code

Before changing a neural-network architecture:

1. **Task:** What exactly must be predicted? Which transformations truly preserve its label?
2. **Data:** Image resolution, channels, sample count, train/test shifts, class balance, and preprocessing?
3. **Model assumptions:** Is locality appropriate? Is weight sharing appropriate? Is absolute position important?
4. **Head:** Does flattening preserve unwanted positional dependence? Would GAP better match the task?
5. **Depth:** Is the receptive field large enough for relevant global patterns?
6. **Normalization:** Does GroupNorm's group count divide channel count? Are skip-connection shapes compatible?
7. **Augmentation:** Does each sampled transformation actually preserve the label, and is its range appropriate?
8. **Evaluation:** Compare unmodified data with shifted, scaled, rotated, and/or permuted inputs only where meaningful. Keep train/test separation intact.
9. **Ablations:** Compare a baseline MLP, an augmented baseline, and a CNN only when the project's constraints permit them.
10. **Interpretation:** Separate representational capacity, optimization dynamics, sample efficiency, and out-of-distribution generalization.

## 9. Suggested prompts for the VS Code assistant

- "Read `Steering_Network_Learning_VSCode_Knowledge.md` before reviewing my PyTorch architecture. Identify the inductive biases it assumes and whether they fit my dataset."
- "Use the lecture's distinction between equivariance and invariance to explain what my pooling and classification head are doing."
- "Check every Conv2d output shape, GroupNorm setting, residual addition, and receptive field in this model."
- "Design an ablation comparing my current model with an augmented variant, without changing any prohibited architectural components."
- "Suggest label-preserving augmentations separately for MNIST, CIFAR-10, and ImageNet, and explicitly flag transformations that might change the label."

## 10. Three takeaways (slide 42)

1. Invariances belong to the task; inductive biases belong to the learner. Learning works best when they match.
2. Augmentation demonstrates invariance through examples; architecture assumes it through computation and weights.
3. The right bias can be worth substantial training data; the wrong bias can steer learning toward worse solutions.
