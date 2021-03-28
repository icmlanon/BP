import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# from quantize import *
from quant_nets import *

interpolate = True
quantize = True
bits_w = 8 
bits_a = 8

class AlexNet_Fine(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet_Fine, self).__init__()
        # q_Conv2d = conv2d_clipped_inputs(10, -10)
        # q_Linear = linear_clipped_inputs(10, -10)
        self.conv1 = Quant_Conv2d_Clipped(3, 64, kernel_size=11, stride=4, padding=5, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a)
        self.actfn1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Quant_Conv2d_Clipped(64, 192, kernel_size=5, padding=2, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a)
        self.actfn2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Quant_Conv2d_Clipped(192, 384, kernel_size=3, padding=1, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a)
        self.actfn3 = nn.ReLU(inplace=True)
        self.conv4 = Quant_Conv2d_Clipped(384, 256, kernel_size=3, padding=1, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a)
        self.actfn4 = nn.ReLU(inplace=True)
        self.conv5 = Quant_Conv2d_Clipped(256, 256, kernel_size=3, padding=1, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a)
        self.actfn5 = nn.ReLU(inplace=True)
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = Quant_IP_Clipped(256, num_classes, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a)

    def forward(self, x):
        # x = self.features(x)
        x = self.conv1(x)
        x = self.actfn1(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.actfn2(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.actfn3(x)

        x = self.conv4(x)
        x = self.actfn4(x)

        x = self.conv5(x)
        x = self.actfn5(x)
        x = self.mp5(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def alexnet_fine(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet_Fine(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model