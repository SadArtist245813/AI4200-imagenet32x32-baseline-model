import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .model import TASKS, ResidualMLP


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, cosine learning-rate decay, no dropout, no augmentation.
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = {"mnist": 10, "cifar10": 30, "imagenet32": 15}


def prepare(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1]."""
    return (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5


def train(task, X_train, y_train, seed=0, device="cuda", record=False):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    # 1. Build the model for this task. Only input size and class count differ.
    model = ResidualMLP(**TASKS[task]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    dataset = TensorDataset(prepare(X_train), torch.as_tensor(y_train).long())
    if record:
        validation_size = max(1, len(dataset) // 10)
        dataset, validation_set = random_split(
            dataset, [len(dataset) - validation_size, validation_size],
            generator=torch.Generator().manual_seed(seed)
        )
        validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Schedualer for learning rate decay: cosine annealing from learning rate to eta_min over the course of training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = EPOCHS[task], eta_min=0.00001 # opyimizer, number of epochs, minimum learning rate
    )
    
    diagnostics = {
        "train_loss": [],
        "epoch_train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "diagnostic_steps": [],
        "epoch_end_steps": [],
        "stream_variance": [],
        "sample_correlation": [],
        "stream_grad_rms": [],
        "linear1_grad_rms": [],   # blocks[i].fc1.weight gradient RMS
        "linear2_grad_rms": [],   # blocks[i].fc2.weight gradient RMS
        "diverged_at_step": None,
        "validation_accuracy": [],
        "test_accuracy": None,    # filled by the harness after evaluation
        "learning_rate": LEARNING_RATE,
    } if record else None
    model.diagnostics = diagnostics

    step = 0  # number of completed optimizer updates

    def record_snapshot(streams):
        # The last epoch's final snapshot may already cover this step.
        if diagnostics["diagnostic_steps"] and diagnostics["diagnostic_steps"][-1] == step:
            return
        diagnostics["diagnostic_steps"].append(step)
        diagnostics["stream_variance"].append([
            stream.detach().var(unbiased=False).item() for stream in streams
        ])
        correlations = []
        for stream in streams:
            samples = stream.detach()
            samples = samples - samples.mean(dim=1, keepdim=True)
            samples = nn.functional.normalize(samples, dim=1)
            batch_size = samples.size(0)
            correlation = (
                (samples.sum(dim=0).square().sum() - samples.square().sum())
                / (batch_size * (batch_size - 1))
                if batch_size > 1 else samples.new_zeros(())
            )
            correlations.append(correlation.item())
        diagnostics["sample_correlation"].append(correlations)
        diagnostics["stream_grad_rms"].append([
            stream.grad.detach().square().mean().sqrt().item()
            for stream in streams
        ])
        diagnostics["linear1_grad_rms"].append([
            block.fc1.weight.grad.detach().square().mean().sqrt().item()
            for block in model.blocks
        ])
        diagnostics["linear2_grad_rms"].append([
            block.fc2.weight.grad.detach().square().mean().sqrt().item()
            for block in model.blocks
        ])

    # 4. The training loop.
    model.train()
    for epoch in range(EPOCHS[task]):
        total_loss = 0.0
        total_correct = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if task in ("cifar10", "imagenet32"):  # training-only flips and padded crops
                flip = torch.rand(len(X_batch), device=X_batch.device) < 0.5
                X_batch[flip] = X_batch[flip].flip(-1)
                padded = nn.functional.pad(X_batch, (4, 4, 4, 4), value=-1.0)
                crops = padded.unfold(2, 32, 1).unfold(3, 32, 1)
                offsets = torch.randint(9, (len(X_batch), 2), device=X_batch.device)
                X_batch = crops[torch.arange(len(X_batch), device=X_batch.device),
                                :, offsets[:, 0], offsets[:, 1]]

            streams = [] if record else None
            logits = model(X_batch, streams=streams)  # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?
            
            # Recoding Diagnostics
            if diagnostics is not None:
                diagnostics["train_loss"].append(loss.item())
                
            optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients

            # Recoding Diagnostics
            if diagnostics is not None:
                record_snapshot(streams)
                streams.clear()              # free memory for the next batch

            optimizer.step()                  # update the weights
            step += 1

            total_loss += loss.item() * len(X_batch)
            if record:
                total_correct += (logits.argmax(dim=1) == y_batch).sum().item()

        print(f"epoch {epoch + 1}: train loss {total_loss / len(dataset):.4f}")

        # Recording Diagnostics
        if diagnostics is not None:
            diagnostics["epoch_end_steps"].append(step)
            # Measure after the final update; do not take another optimizer step.
            streams = []
            logits = model(X_batch, streams=streams)
            optimizer.zero_grad()
            loss_fn(logits, y_batch).backward()
            record_snapshot(streams)
            streams.clear()

            # Evaluate held-out training examples without augmentation or updates.
            model.eval()
            validation_loss, validation_correct = 0.0, 0
            with torch.no_grad():
                for X_val, y_val in validation_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    val_logits = model(X_val)
                    validation_loss += loss_fn(val_logits, y_val).item() * len(y_val)
                    validation_correct += (val_logits.argmax(dim=1) == y_val).sum().item()
            diagnostics["epoch_train_loss"].append(total_loss / len(dataset))
            diagnostics["train_accuracy"].append(total_correct / len(dataset))
            diagnostics["validation_loss"].append(validation_loss / len(validation_set))
            diagnostics["validation_accuracy"].append(validation_correct / len(validation_set))
            print(f"  validation loss {diagnostics['validation_loss'][-1]:.4f}, "
                  f"accuracy {diagnostics['validation_accuracy'][-1]:.2%}")
            model.train()

        scheduler.step()  # update the learning rate for the next epoch

    return model


def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)
