import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from quant_nets import *


__all__ = ['Quant_AlexNet', 'quant_alexnet']


interpolate = True
quantize = True
bits_w = 8
bits_a = 8

class Quant_AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(Quant_AlexNet, self).__init__()
        self.features = nn.Sequential(
            Quant_Conv2d(3, 64, kernel_size=11, stride=4, padding=2,quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Quant_Conv2d(64, 192, kernel_size=5, padding=2,quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Quant_Conv2d(192, 384, kernel_size=3, padding=1,quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            Quant_Conv2d(384, 256, kernel_size=3, padding=1,quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            Quant_Conv2d(256, 256, kernel_size=3, padding=1,quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            Quant_IP(256 * 6 * 6, 4096,quantize=quantize, interpolate=interpolate),
            #nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Quant_IP(4096, 4096,quantize=quantize, interpolate=interpolate),
            #nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            Quant_IP(4096, num_classes,quantize=quantize, interpolate=interpolate),
            #nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def quant_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Quant_AlexNet(**kwargs)
    return model
