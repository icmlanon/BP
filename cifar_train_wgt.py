'''Script for targeted training of bitlengths and weights with a weighted loss function of cifar10 models.'''

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


"""Select loss function here. Weighted loss function for targeting specific criteria"""
#wgts=[23232,307200,663552,884736,589824,2560] #AlexNet Mem
#acts=[3072,1024,768,1536,1024,256]# AlexNet Mem

#wgts=[12288,786432,1179648,393216,262144,2560] #AlexNet MACs
#acts=[12288,786432,1179648,393216,262144,2560] #AlexNet MACs

#wgts=[1728,36864,36864,36864,36864,73728,147456,8192,147456,147456,294912,589824,32768,589824,589824,1179648,2359296,131072,2359296,2359296,5120] #ResNet Mem
#acts=[3072,65536,65536,65536,65536,65536,32768,32768,32768,32768,32768,16384,16384,16384,16384,16384,8192,8192,8192,8192,512] #ResNet Mem
wgts=[196608,4194304,4194304,4194304,4194304,8388608,1048576,1048576,4194304,4194304,8388608,1048576,1638400,4194304,4194304,8388608,1048576,2359296,4194304,4194304,5120] #ResNet MACs
acts=[196608,4194304,4194304,4194304,4194304,8388608,1048576,1048576,4194304,4194304,8388608,1048576,1638400,4194304,4194304,8388608,1048576,2359296,4194304,4194304,5120] #ResNet MACs
#acts = [x * 128 for x in acts]
loss_func = Quant_Loss_Weighted(model,wgts=wgts,acts=acts,reg_strength=0.01)
#loss_func=Quant_Loss(model=model,reg_strength=0.01)
#loss_func = nn.CrossEntropyLoss()



learn = Learner(data, model.cuda(), metrics=[accuracy], loss_func=loss_func)
learn.callback_fns += [partial(TrackEpochCallback), partial(SaveModelCallback, every='epoch', name='model')]

"""Train:"""
print_model_mod(model)
learn.fit_one_cycle(300)
#learn.fit(500)
print_model_mod(model)
#learn.save('alex')
#learn.fit_one_cycle(35, 3e-3, wd=0.4)
