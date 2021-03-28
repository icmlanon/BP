"""Print bitlengths after each epoch"""
from fastai.script import *
from fastai.vision import *
from fastai.distributed import *
from fastai.callbacks.tracker import *

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from quant_nets import *
from quant_resnets import *
from quant_alexnet import *
from quant_densenet import *
from cifar_quant_mobilenetv2 import *
from alex import *
from resnets import *
from quant_dpn import *
from quant_resnext import *
from quant_preact_resnet import *

from models.preact_resnet import *
from models.densenet import *
from models.resnext import *
from models.mobilenetv2 import *
from models.dpn import *

import argparse

models_dict = {
  "alexnet": alexnet(num_classes=10),
  "resnet18": ResNet18(num_classes=10),
  "pre_act_resnet18": PreActResNet18(),
  "densenet121": DenseNet121(),
  "resnext29": ResNeXt29_2x64d(),
  "mobilenetV2": MobileNetV2(),
  "DPN92": DPN92(),
  "bp_alexnet": quant_alexnet(num_classes=10),
  "bp_resnet18": Quant_ResNet18(),
  "bp_pre_act_resnet18": Quant_PreActResNet18(),
  "bp_densenet121": Quant_DenseNet121(),
  "bp_resnext29": Quant_ResNeXt29_2x64d(),
  "bp_mobilenetV2": Quant_MobileNetV2(),
  "bp_DPN92": Quant_DPN92()
  
  }
  
parser = argparse.ArgumentParser(description='Train on cifar10')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=models_dict.keys(),
                    help='model architecture: ' +
                    ' | '.join(models_dict.keys()) +
                    ' (default: alexnet)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')

args = parser.parse_args()
path = untar_data(URLs.CIFAR)#_SAMPLE)

stats=cifar_stats
data = ImageDataBunch.from_folder(path).normalize(stats)

"""Select model"""
model=models_dict[args.model]



loss_func = Quant_Loss(model=model,reg_strength=0.01)
learn = Learner(data, model.cpu(), metrics=[accuracy], loss_func=loss_func)

"""Load trained weights"""
for i in range(1,args.epochs):
	learn.load('model_'+str(i))
	print_model_mod(learn.model)
	



