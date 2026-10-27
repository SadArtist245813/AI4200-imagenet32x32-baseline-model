### Change 0 — mnist — 2026-10-8

**Change:** This the orginal run with the orginal code.
**Why:** I wanted to see where to start and find and make sure I could run the program.
**Expected:** I expected it to produce results but for it not to be the best due to one simple obvious issue, the tanh activation funtion.
**Result:** Accuracy 95.06%. Training time 0.5 min. Curve is not shown.

### Change 0 — cifar10 — 2026-10-8

**Change:** This the orginal run with the orginal code.
**Why:** I wanted to see where to start and find and make sure I could run the program.
**Expected:** I expected it to produce results but for it not to be the best due to one simple obvious issue, the tanh activation funtion.
**Result:** Accuracy 48.61%. Training time 1.3 min. Curve is not shown.

### Change 0 — imagenet32 — 2026-10-8

**Change:** This the orginal run with the orginal code.
**Why:** I wanted to see where to start and find and make sure I could run the program.
**Expected:** I expected it to produce results but for it not to be the best due to one simple obvious issue, the tanh activation funtion.
**Result:** Accuracy 6.77%. Training time 12.4 min. Curve is not shown.

### Change 1 — mnist — 2026-10-13

**Change:** I change the activation function to ReLU.
**Why:** It was clear as day that the activation function need a change.
**Expected:** I expected the change to be noticable, but not by much, I need a good foudation to start on.
**Result:** Accuracy 95.06% -> 97.11%. Training time 0.5 min -> 0.9 min. Curve is not shown.

### Change 1 — cifar10 — 2026-10-13

**Change:** I change the activation function to ReLU.
**Why:** It was clear as day that the activation function need a change.
**Expected:** I expected the change to be noticable, but not by much, I need a good foudation to start on.
**Result:** Accuracy 48.61% -> 51.59%. Training time 1.3 min -> 2.3 min. Curve is not shown.

### Change 1 — imagenet32 — 2026-10-13

**Change:** I change the activation function to ReLU.
**Why:** It was clear as day that the activation function need a change.
**Expected:** I expected the change to be noticable, but not by much, I need a good foudation to start on.
**Result:** Accuracy 6.77% -> 8.04%. Training time 12.4 min -> 20 min. Curve is not shown.

### Change 2 — mnist — 2026-10-24

**Change:** I change the activation function to SiLU. I added graph support to help with analyzation.
**Why:** I want to find how how far I could go with my activation funtion before I had to focus on another known problem, the model size.
**Expected:** Very similar or worse then the orginal accuracy.
**Result:** Accuracy 97.11% -> 96.99%. Training time 0.9 min -> 1.0 min. Very similar or worse then the orginal accuracy.

### Change 2 — cifar10 — 2026-10-24

**Change:** I change the activation function to SiLU. I added graph support to help with analyzation.
**Why:** I want to find how how far I could go with my activation funtion before I had to focus on another known problem, the model size.
**Expected:** Very similar or worse then the orginal accuracy.
**Result:** Accuracy 51.59% -> 49.15%. Training time 2.3 min -> 2.5 min. Loss keeps decreasing but becomes noisy.

### Change 2 — imagenet32 — 2026-10-24

**Change:** I change the activation function to SiLU. I added graph support to help with analyzation.
**Why:** I want to find how how far I could go with my activation funtion before I had to focus on another known problem, the model size.
**Expected:** Very similar or worse then the orginal accuracy.
**Result:** Accuracy 8.04% -> 8.42%. Training time 20 min -> 23.1 min. Curve is not shown.

### Change 3 — mnist — 2026-10-25

**Change:** I change the size off the network 15 block.
**Why:** My graphs I added in my last change were pointing to a need of different sizes in the network.
**Expected:** The model to be more accurate, but not change too much due to the small size and needs of the dataset.
**Result:** Accuracy 96.99% -> 97.71%. Training time 1.0 min -> 0.8 min. Curves in ./graphs/mnist-change-3.png

### Change 3 — cifar10 — 2026-10-25

**Change:** I change the activation function to SiLU. I added graph support to help with analyzation.
**Why:** My graphs I added in my last change were pointing to a need of different sizes in the network.
**Expected:** The model to get more accurate with a smaller size since I have seemed to learn noise.
**Result:** Accuracy 49.15% -> 49.93%. Training time 2.5 min -> 1.9 min. Curves in ./graphs/cifar10-change-3.png

### Change 3 — imagenet32 — 2026-10-25

**Change:** I change the number of block to 15. I changed the number of epochs to 15
**Why:** My graphs I added in my last change were pointing to a need of different sizes in the network.
**Expected:** The network to get more accurate as the network gets bigger. I am afraid of learning noise though leaving me to maybe think of using other types of changes in the model.
**Result:** Accuracy 8.42% -> 9.32%. Training time 23.1 min -> 25.9 min. Curves in ./graphs/imagenet32-change-3.png

### Change 4 — mnist — 2026-10-26

**Change:** I change the activation function to SwiGLU.
**Why:** I thought I might try a diffrent type of structure for curiosity reasons.
**Expected:** Not much of a change other then less noise being learned.
**Result:** Accuracy 97.71% -> 93.14%. Training time 0.8 min -> 0.7 min. Curves in ./graphs/mnist-change-4.png

### Change 4 — cifar10 — 2026-10-26

**Change:** I change the activation function to SwiGLU.
**Why:** I thought I might try a diffrent type of structure for curiosity reasons.
**Expected:** Not much of a change other then less noise being learned.
**Result:** Accuracy 49.93% -> 49.91%. Training time 1.9 min -> 1.7 min. Curves in ./graphs/cifar10-change-4.png

### Change 4 — imagenet32 — 2026-10-26

**Change:** I change the activation function to SwiGLU.
**Why:** I thought I might try a diffrent type of structure for curiosity reasons.
**Expected:** The network to get more accurate as the network learns. I am afraid of learning noise though leaving me to maybe think of using other types of changes in the model.
**Result:** Accuracy 9.32% -> 8.68%. Training time 25.9 min -> 23.7 min. Curves in ./graphs/imagenet32-change-4.png

### Change 5 — mnist — 2026-10-26

**Change:** I add a cosine learning rate decay scheduler with a minimum of 0.00001.
**Why:** Will suggested to add a scheduler and AdamW. It would add better control too.
**Expected:** The network to go back to the better accuracy.
**Result:** Accuracy 93.14% -> 98.51%. Training time 0.7 min -> 0.8 min. Curves in ./graphs/mnist-change-5.png

https://docs.pytorch.org/docs/main/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html

### Change 5 — cifar10 — 2026-10-26

**Change:** I add a cosine learning rate decay scheduler with a minimum of 0.00001.
**Why:** Will suggested to add a scheduler and AdamW. It would add better control too.
**Expected:** The accuracy to be a little better, I most like would need to look as some my parameters to help discover better methods.
**Result:** Accuracy 49.91% -> 54.69%. Training time 1.7 min -> 2.1 min. Curves in ./graphs/cifar10-change-5.png

https://docs.pytorch.org/docs/main/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html

### Change 5 — imagenet32 — 2026-10-26

**Change:** I add a cosine learning rate decay scheduler with a minimum of 0.00001.
**Why:** Will suggested to add a scheduler and AdamW. It would add better control too.
**Expected:** Not too much of a change. I most likey depends on my parameters.
**Result:** Accuracy 8.68% -> 8.76%. Training time 23.7 min -> 31.5 min. Curves in ./graphs/imagenet32-change-5.png

https://docs.pytorch.org/docs/main/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html

### Change 6 — mnist — 2026-10-26

Though some changes do take affect, they all seem to have an extemely similar resualt of 95%-98% accuracy. The resaults are not noting.

**Change:** Blocks to 10.

### Change 6 — cifar10 — 2026-10-26

**Change:** I add Accuracy and Validation Accuracy graph. I added randomized flipping to the images for input.
**Why:** I needed an overview for the obviouse overfitting. I also want a way to help with overfitting so I added image flipping. It changes random images in the dataset to be flipped.
**Expected:** The overfitting to be more fixed then before, but not completely.
**Result:** Accuracy 54.69% -> 57.48%. Training time 2.1 min -> 1.5. Curves in ./graphs/cifar10-change-6-test-1.png

### Change 6 — imagenet32 — 2026-10-26

I keep this one untested so I can produce resaults faster for now. It will use the same technique as cifar10 soon.

### Change 7 — mnist — 2026-10-26

Though some changes do take affect, they all seem to have an extemely similar resualt of 95%-98% accuracy. The resaults are not noting.

### Change 7 — cifar10 — 2026-10-26

**Change:** I added 4-pixel padded random croping.
**Why:** Just another way to mess with the dataset. It added a padded border and crops a random part of the image. 32x32 -> 40x40 -> 32x32
**Expected:** It to be less overfitted.
**Result:** Accuracy 57.48% -> 58.52%. Training time 2.1 min -> 2.1. Curves in ./graphs/cifar10-change-7.png

### Change 7 — imagenet32 — 2026-10-26

I keep this one untested so I can produce resaults faster for now. It will use the same technique as cifar10 soon.

### Change 8 — mnist — 2026-10-26

Though some changes do take affect, they all seem to have an extemely similar resualt of 95%-98% accuracy. The resaults are not noting.


### Change 8 — cifar10 — 2026-10-26

**Change:** 
**Why:** 
**Expected:** 
**Result:** Accuracy 57.48% -> 58.52%. Training time 2.1 min -> 2.1. Curves in ./graphs/cifar10-change-7.png

### Change 8 — imagenet32 — 2026-10-26

I keep this one untested so I can produce resaults faster for now. It will use the same technique as cifar10 soon.