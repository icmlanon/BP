import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from quant_nets import *

interpolate = True
quantize = True
bits_w = 8 
bits_a = 8


k=[1,1,1,1,0.25]

#kd=[4,1,1,1,1]

class AlexNet_Quant(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet_Quant, self).__init__()
        self.features = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            Quant_Conv2d(3, int(64*k[0]), kernel_size=11, stride=4, padding=5, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(64, 192, kernel_size=5, padding=2),
            Quant_Conv2d(int(k[0]*64), int(192*k[1]), kernel_size=5, padding=2, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(192, 384, kernel_size=3, padding=1),
            Quant_Conv2d(int(k[1]*192), int(384*k[2]), kernel_size=3, padding=1, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            #nn.Conv2d(384, 256, kernel_size=3, padding=1),
            Quant_Conv2d(int(k[2]*384), int(256*k[3]), kernel_size=3, padding=1, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            Quant_Conv2d(int(k[3]*256), int(256*k[4]), kernel_size=3, padding=1, quantize=quantize, interpolate=interpolate, bits_w=bits_w, bits_a=bits_a),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = Quant_IP(int(256*k[4]), num_classes)
        #self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def quant_alexnet_bin(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet_Quant(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

















