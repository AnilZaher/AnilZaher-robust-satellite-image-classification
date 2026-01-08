import os
import types
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.data_utils import SafeImageFolder



# plots the metrics of the model after traning
def plot_training_metrics(train_losses, val_losses,train_acc, val_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(train_losses)+1)
    # subplot 1: Loss
    ax1.plot(epochs, train_losses, label="Train Loss Per Epoch", color='blue')
    ax1.plot(epochs, val_losses, label="Validation Loss Per Epoch", color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel("Mean Epoch Loss")
    ax1.set_title(f"Loss vs Epoch")
    ax1.grid(True)
    ax1.legend()

    # subplot 2: Accuracy
    ax2.plot(epochs, train_acc, label="Train Accuracy Per Epoch", color='green')
    ax2.plot(epochs, val_acc, label="Validation Accuracy Per Epoch", color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel("Accuracy [%]")
    ax2.set_title(f"Accuracy vs Epoch")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# denormalizes single images (not batches) and prepares them for im_show formate
def convert_to_imshow_format(image):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)

    image = image.detach()
    image = image * std + mean
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()
    return image

# permutes the images so its compatible with im_show input format
def to_imshow(x):
    return x.permute(1, 2, 0).numpy()

# visualization the classed of the data
def class_viz(split_root):
    viz_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    images_root = os.path.join(split_root,'train')
    dataset = SafeImageFolder(images_root, transform=viz_transform)

    # collecting one image per class
    images_per_class = {}
    class_names = dataset.classes
    num_classes = len(class_names)

    for img, label in dataset:
        if label not in images_per_class:
            images_per_class[label] = img
        if len(images_per_class) == num_classes:
            break

    # creating a list of images
    images = [images_per_class[i] for i in range(num_classes)]
    labels = list(range(num_classes))

    # plotting the images
    fig, axes = plt.subplots(2, num_classes // 2, figsize=(10, 2.5))
    axes = axes.flatten()

    for idx, image in enumerate(images):
        axes[idx].imshow(to_imshow(image))
        axes[idx].set_title(class_names[labels[idx]])
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.tight_layout()
    plt.show()

# denormalizes the batch using ImageNet mean and std
def denormalize_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)

# renormalizes the batch using ImageNet mean and std
def renormalize_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std



# changing brightness level
def adjust_brightness(x, factor):
    return torch.clamp(x * factor, 0.0, 1.0)

# makes the image hazy
def add_haze(x, alpha=0.5):
    haze = torch.ones_like(x)
    return torch.clamp((1 - alpha) * x + alpha * haze, 0.0, 1.0)

# used for evaluating the model under different severity levels
def evaluate(model, dataloader, device, noise_fn, severity):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if noise_fn == None:
                noisy_x = x
            else:
                # go to image space
                x_img = denormalize_batch(x)

                # apply perturbation
                noisy_img = noise_fn(x_img, severity)

                # go back to normalized space
                noisy_x = renormalize_batch(noisy_img)

            outputs = model(noisy_x)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return (correct / total) * 100


# grad-cam for renet 50
def gradcam(model, input_tensor, class_idx=None):
    model.eval()
    target_layers = [model.layer4[-1]]

    # if no class specified, use predicted class
    if class_idx is None:
        with torch.no_grad():
            outputs = model(input_tensor)
            class_idx = outputs.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(class_idx)]

    # running Grad-CAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # de-normalize input image (because we used ImageNet's normalization and initial weights)
    image = input_tensor[0].detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()

    # overlay on top of a heatmap
    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True, image_weight=0.9)

    return cam_image


# attention maps for dinov3 and vit_base
def attention_map(model, input_tensor, model_type, patch_size=16):
    model.eval()
    attn_holder = []

    if model_type == "dinov3":
        # get attention directly from hf dino
        attn = model.get_last_selfattention(input_tensor)

    elif model_type == "vit_base":
        # force torchvision vit to return attention weights
        layer = model.encoder.layers[-1].self_attention
        orig_forward = layer.forward

        # patch forward to always request attention weights
        layer.forward = types.MethodType(
            lambda self, q, k, v, **kw:
            orig_forward(q, k, v,need_weights=True,**{k: v for k, v in kw.items() if k != "need_weights"}),layer)

        # capture attention weights from forward output
        handle = layer.register_forward_hook(
            lambda _m, _i, output: attn_holder.append(output[1]))

        with torch.no_grad():
            _ = model(input_tensor)

        handle.remove()
        layer.forward = orig_forward

        if len(attn_holder) == 0:
            raise RuntimeError("attention extraction failed for vit_base")

        attn = attn_holder[0]

    else: # cant use attention maps with resnet50
        raise ValueError("model_type must be 'dinov3' or 'vit_base'")

    # average over heads if needed
    if attn.ndim == 4:
        attn = attn.mean(dim=1)

    # extract cls to patch attention
    if model_type == "dinov3":
        num_reg = model.backbone.config.num_register_tokens
        cls_attn = attn[:, 0, 1 + num_reg :]
    else:
        cls_attn = attn[:, 0, 1:]

    p = torch.quantile(cls_attn, 0.95)
    cls_attn = torch.clamp(cls_attn / p, 0, 1)

    # reshape patch tokens to grid
    num_patches = cls_attn.shape[-1]
    h = w = int(num_patches ** 0.5)

    if h * w != num_patches:
        raise RuntimeError(f"invalid patch count: {num_patches}")

    cls_attn = cls_attn.reshape(1, 1, h, w)

    # upsample to image resolution
    attn_map = F.interpolate(cls_attn,size=(input_tensor.shape[2], input_tensor.shape[3]),mode="bilinear",align_corners=False)[0, 0].cpu().numpy()

    # denormalize image for visualization
    img = input_tensor[0].detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    return show_cam_on_image(img, attn_map, use_rgb=True, image_weight=0.9)



# bar plots for accuracy vs perturbation
def plot_accuracy_vs_severity(acc_dict, perturbation_name, x_label, x_values):
    models = list(acc_dict.keys())
    num_models = len(models)

    x = np.arange(len(x_values))
    width = 0.8 / num_models

    fig, ax = plt.subplots(figsize=(10, 5))

    # creating the bars
    for i, model in enumerate(models):
        accs = acc_dict[model]
        ax.bar(x + i * width, accs, width, label=model.upper())

    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy (%)")
    #ax.set_title(f"Accuracy under {perturbation_name} perturbation") # no plot title needed for the report (we added captions instead)
    ax.set_xticks(x + width * (num_models - 1) / 2)
    ax.set_xticklabels([f"{v}" for v in x_values])
    ax.set_yticks(np.arange(0, 101, 5))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

