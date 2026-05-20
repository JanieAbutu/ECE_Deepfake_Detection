
# ================================================================
# MODEL USED: EfficientNet-B7 (FINETUNED)
# ================================================================
class EfficientNetB7Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b7(weights="IMAGENET1K_V1")


        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1)  
        )

    def forward(self, x):
        return self.backbone(x)

model = EfficientNetB7Binary().to(DEVICE)
print(model)
