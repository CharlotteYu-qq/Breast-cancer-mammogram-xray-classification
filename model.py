import torch
import torch.nn as nn
import torchvision.models as models

class BreastModel(nn.Module):
    def __init__(self, backbone='efficientnet_b0', num_classes=3):
        super(BreastModel, self).__init__()

        # use new weights API for EfficientNet
        if backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b0(weights=weights)
        elif backbone == 'efficientnet_b1':
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b1(weights=weights)
        elif backbone == 'efficientnet_b2':
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b2(weights=weights)
        elif backbone == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b3(weights=weights)
        elif backbone == 'efficientnet_b4':
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b4(weights=weights)
        elif backbone == 'efficientnet_b5':
            weights = models.EfficientNet_B5_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b5(weights=weights)
        elif backbone == 'efficientnet_b6':
            weights = models.EfficientNet_B6_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b6(weights=weights)
        elif backbone == 'efficientnet_b7':
            weights = models.EfficientNet_B7_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b7(weights=weights)
        else:
            # 默认使用efficientnet_b0
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b0(weights=weights)

        # modify first conv layer to accept single-channel input
        original_conv = self.model.features[0][0]  # EfficientNet's first conv layer
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,  # single-channel input
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # modify last fully connected layer for 3-class classification
        # EfficientNet use classifier not fc
        original_classifier = self.model.classifier[1]
        self.model.classifier[1] = nn.Linear(
            original_classifier.in_features, 
            num_classes
        )

    def forward(self, x):
        return self.model(x)







# import torch
# import torch.nn as nn
# import torchvision.models as models

# class BreastModel(nn.Module):
#     def __init__(self, backbone='resnet18', num_classes=3):
#         super(BreastModel, self).__init__()

#         # use new weights API
#         if backbone == 'resnet18':
#             weights = models.ResNet18_Weights.IMAGENET1K_V1
#             self.model = models.resnet18(weights=weights)
#         elif backbone == 'resnet34':
#             weights = models.ResNet34_Weights.IMAGENET1K_V1
#             self.model = models.resnet34(weights=weights)
#         else:
#             weights = models.ResNet50_Weights.IMAGENET1K_V1
#             self.model = models.resnet50(weights=weights)

#         # modify first conv layer to accept single-channel input
#         original_conv = self.model.conv1
#         self.model.conv1 = nn.Conv2d(
#             in_channels=1,  # single-channel input
#             out_channels=original_conv.out_channels,
#             kernel_size=original_conv.kernel_size,
#             stride=original_conv.stride,
#             padding=original_conv.padding,
#             bias=original_conv.bias is not None
#         )

#         # modify last fully connected layer for 3-class classification
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)