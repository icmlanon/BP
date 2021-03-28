import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from quant_nets import *

interpolate = True
quantize = True
bits_w = 8 
bits_a = 8

class AlexNet_Quant_Clipped(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet_Quant_Clipped, self).__init__()
        self.features = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            Quant_Conv2d_Clipped(3, 64, kernel_size=11, stride=4, padding=5, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(64, 192, kernel_size=5, padding=2),
            Quant_Conv2d_Clipped(64, 192, kernel_size=5, padding=2, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(192, 384, kernel_size=3, padding=1),
            Quant_Conv2d_Clipped(192, 384, kernel_size=3, padding=1, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            #nn.Conv2d(384, 256, kernel_size=3, padding=1),
            Quant_Conv2d_Clipped(384, 256, kernel_size=3, padding=1, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            Quant_Conv2d_Clipped(256, 256, kernel_size=3, padding=1, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = Quant_IP_Clipped(256, num_classes, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a)
        #self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def quant_alexnet_clip(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet_Quant_Clipped(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

















