"""Script for validating a trained cifar model"""
from fastai.distributed import *
from fastai.callbacks.tracker import *

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from quant_nets import *
from quant_resnets import *
from quant_alexnet import *
from quant_alexnet_bin import *
from alex import *
from resnets import *


torch.cuda.set_device(1)
path = untar_data(URLs.CIFAR)
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
stats=cifar_stats
#data = ImageDataBunch.from_folder(path, ds_tfms=ds_tfms, bs=128).normalize(stats)
data = ImageDataBunch.from_folder(path, ds_tfms=ds_tfms, bs=1).normalize(stats)

"""Select model here:"""
#data = ImageDataBunch.from_folder(path)
#model = models.resnet18#resnet18(num_classes=10)#alexnet(num_classes=10) 
#model = resnet18(num_classes=10) 
#model = ResNet18(num_classes=10) 
#model = models.xresnet18(num_classes=10) 
#model = alexnet(num_classes=10) 

#model = quant_alexnet(num_classes=10) 

#model = quant_alexnet_bin(num_classes=10) 
#model = resnet18(num_classes=10) 
#model = ResNet18() 

model = Quant_ResNet18() 



"""Select loss function here"""
loss_func = nn.CrossEntropyLoss()
#loss_func = Quant_Loss(model=model,reg_strength=0.01)
#loss_func = Quant_Loss_Fixed(model=model,reg_strength=0.01)
learn = Learner(data, model.cuda(), metrics=[accuracy], loss_func=loss_func)
learn.callback_fns += [partial(TrackEpochCallback), partial(SaveModelCallback, every='epoch', name='model')]

#learn.load('/home/datasets/cifar10/trained_models/quant_alex_25_fin')

"""Load trained model"""
#learn.load('/home/datasets/cifar10/trained_models/quant_res_10_fix')
learn.load('/home/datasets/cifar10/models_res_10_final/model_299')
#learn.load('/home/datasets/cifar10/trained_models/quant_alex_40')


for m in learn.model.modules():
    if isinstance(m, Quant_Conv2d) or isinstance(m, Quant_IP):
        print(m.alpha_w.data,m.alpha_a.data)
        m.alpha_w.data=np.ceil(m.alpha_w.data).cuda()
        m.alpha_a.data=np.ceil(m.alpha_a.data).cuda()
        m.alpha_w.trainable = False
        m.alpha_a.trainable = False
        m.alpha_w.requires_grad = False
        m.alpha_a.requires_grad = False

learn.loss_func = nn.CrossEntropyLoss()
print_model(learn.model)
#learn.save('model_0')
#learn.load('model_0')

print_model_weights(learn.model)
#learn.data.valid_ds=learn.data.valid_ds[[0]]
#print(len(learn.data.valid_ds))
#quit()

"""Validation"""
torch.cuda.set_device(1)
print(learn.validate(metrics=[accuracy]))
#learn.fit(500)
print_model_mod(learn.model)
#learn.save('alex')
#learn.fit_one_cycle(35, 3e-3, wd=0.4)
