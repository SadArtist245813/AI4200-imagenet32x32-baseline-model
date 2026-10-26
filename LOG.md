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
**Result:** Accuracy 95.06% -> 97.11%. Training time 0.5 min -> 0.9. Curve is not shown.

### Change 1 — cifar10 — 2026-10-13

**Change:** I change the activation function to ReLU.
**Why:** It was clear as day that the activation function need a change.
**Expected:** I expected the change to be noticable, but not by much, I need a good foudation to start on.
**Result:** Accuracy 48.61% -> 51.59%. Training time 1.3 min -> 2.3. Curve is not shown.

### Change 1 — imagenet32 — 2026-10-13

**Change:** I change the activation function to ReLU.
**Why:** It was clear as day that the activation function need a change.
**Expected:** I expected the change to be noticable, but not by much, I need a good foudation to start on.
**Result:** Accuracy 6.77% -> 8.04%. Training time 12.4 min -> 20. Curve is not shown.

### Change 2 — mnist — 2026-10-24

**Change:** I change the activation function to SiLU. I added graph support to help with analyzation.
**Why:** I want to find how how far I could go with my activation funtion before I had to focus on another known problem, the model size.
**Expected:** Very similar or worse then the orginal accuracy.
**Result:** Accuracy 97.11% -> 96.99. Training time 0.9 -> 1.0>. Very similar or worse then the orginal accuracy.

### Change 2 — cifar10 — 2026-10-24

**Change:** I change the activation function to SiLU. I added graph support to help with analyzation.
**Why:** I want to find how how far I could go with my activation funtion before I had to focus on another known problem, the model size.
**Expected:** Very similar or worse then the orginal accuracy.
**Result:** Accuracy 51.59% -> 49.15. Training time 2.3 -> 2.5. Loss keeps decreasing but becomes noisy.

### Change 2 — imagenet32 — 2026-10-24

**Change:** I change the activation function to SiLU. I added graph support to help with analyzation.
**Why:** I want to find how how far I could go with my activation funtion before I had to focus on another known problem, the model size.
**Expected:** Very similar or worse then the orginal accuracy.
**Result:** Accuracy 8.04% -> 8.42. Training time 20 -> 23.1. Curve is not shown.