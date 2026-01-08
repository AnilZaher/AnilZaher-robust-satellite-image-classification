import torch.nn as nn
from torchvision import models
from transformers import AutoModel


def get_model(name, num_classes=10,fine_tune=False):
    if name == "resnet50":
        # using ImageNet weights
        model = models.resnet50(weights='IMAGENET1K_V2')
        if fine_tune == False: # only linear probing
            for param in model.parameters(): param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "vit_base":
        # using ImageNet weights
        model = models.vit_b_16(weights='IMAGENET1K_V1')
        if fine_tune == False:
            for param in model.parameters(): param.requires_grad = False
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif name == "dinov3":
        # DINOv3 ViT-B pre-trained on LVD-1689M
        backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m", output_attentions=True)
        for param in backbone.parameters(): param.requires_grad = False # no fine-tuning with dino because its hugh

        class DINOv3Classifier(nn.Module):
            def __init__(self, backbone, num_classes):
                super().__init__()
                self.backbone = backbone # feature extractor
                self.classifier = nn.Linear(768, num_classes)  # DINOv3 embedding size

            def forward(self, x):
                # using the [CLS] token as a global representation
                outputs = self.backbone(x)
                return self.classifier(outputs.last_hidden_state[:, 0, :])

            def get_last_selfattention(self, x):
                outputs = self.backbone(x, output_attentions=True)
                # attentions is a tuple: one tensor per layer
                # each tensor: (B, heads, tokens, tokens)
                return outputs.attentions[-1]

        model = DINOv3Classifier(backbone, num_classes)

    return model