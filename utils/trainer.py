import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# training loop
def train(model, loader, criterion, optimizer, device, leave = False):
    model.train()

    running_loss = 0.0
    running_correct = 0

    for inputs, labels in tqdm(loader, desc="Training", leave=leave):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100.0 * running_correct / len(loader.dataset)
    return epoch_loss, epoch_acc


# validation loop
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100.0 * running_correct / len(loader.dataset)
    return epoch_loss, epoch_acc
