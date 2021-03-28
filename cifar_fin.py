'''Script to fine-tune a cifar10 model after the integer bitlengths have been selected'''


from fastai.script import *
from fastai.vision import *
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
data = ImageDataBunch.from_folder(path, ds_tfms=ds_tfms, bs=128).normalize(stats)
#data = ImageDataBunch.from_folder(path)


"""Select model here:"""
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


learn = Learner(data, model.cuda(), metrics=[accuracy], loss_func=loss_func)
learn.callback_fns += [partial(TrackEpochCallback), partial(SaveModelCallback, every='epoch', name='model')]

"""Load non-int bitlength model:"""
#learn.load('/home/datasets/cifar10/trained_models/cifar_alex_05')
learn.load('/home/datasets/cifar10/trained_models/squared_res_10')
#learn.load('/home/datasets/cifar10/trained_models/cifar_res_25')
#learn.load('/home/datasets/cifar10/trained_models/quant_10')


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


"""Train:"""
#learn.fit_one_cycle(60, 5e-3, wd=0.0)
learn.fit(60, 5e-4, wd=0.0)
