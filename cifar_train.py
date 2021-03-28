'''Script for training the bitlengths and weights for cifar10 models'''


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
parser.add_argument('--gpu', type=int,default='0',
                    help='gpu used for training (default: 0)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--loss', default='BP', type=str, metavar='LOSS',
                    help='loss function used (XE or BP)')
parser.add_argument('--r', '--regularizer', default=1.0, type=float,
                    metavar='R', help='regularizer (default: 1.0)')

args = parser.parse_args()


print(args.model,args.gpu,args.epochs,args.batch,args.loss,args.r)

#select GPU:
torch.cuda.set_device(args.gpu)

#set up the dataset
path = untar_data(URLs.CIFAR)
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
stats=cifar_stats
data = ImageDataBunch.from_folder(path, ds_tfms=ds_tfms, bs=args.batch).normalize(stats)

#select the model
model=models_dict[args.model]

#select the loss function
if (args.loss=="XE"):
	loss_func = nn.CrossEntropyLoss()
elif (args.loss=="BP"):
	loss_func = Quant_Loss(model=model,reg_strength=args.r)


learn = Learner(data, model.cuda(), metrics=[accuracy], loss_func=loss_func)
#learn.callback_fns += [partial(TrackEpochCallback), partial(SaveModelCallback, every='epoch', name='model')]
learn.callback_fns += [partial(SaveModelCallback, every='epoch', name='model')]
"""Train:"""
print_model_mod(model)
learn.fit_one_cycle(args.epochs)
print_model_mod(model)
