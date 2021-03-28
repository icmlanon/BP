from fastai.script import *
from fastai.vision import *
from fastai.distributed import *
from fastai.callbacks.tracker import *

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from quant_nets import *
from quant_resnets import *
from quant_alexnet import *
from quant_alexnet_individual import *
from quant_alexnet_bin import *
from quant_alexnet_pact import *
from alex import *
from resnets import *

#torch.cuda.set_device(0)
path = untar_data(URLs.CIFAR)
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
stats=cifar_stats
data = ImageDataBunch.from_folder(path, ds_tfms=ds_tfms, bs=128).normalize(stats)
#data = ImageDataBunch.from_folder(path)
#model = models.resnet18#resnet18(num_classes=10)#alexnet(num_classes=10) 
#model = resnet18(num_classes=10) 
#model = ResNet18(num_classes=10) 
#model = models.xresnet18(num_classes=10) 
#model = alexnet(num_classes=10) 
model = quant_alexnet(num_classes=10) 
#model = quant_alexnet_PACT(num_classes=10) 
#model = quant_alexnet_bin(num_classes=10) 
#model = quant_alexnet_individual(num_classes=10) 

#model = resnet18(num_classes=10) 
#model = ResNet18() 
#model = Quant_ResNet18() 




loss_func = Quant_Loss(model=model,reg_strength=0.01)
#loss_func = nn.CrossEntropyLoss()
learn = Learner(data, model.cuda(), metrics=[accuracy], loss_func=loss_func)
learn.callback_fns += [partial(TrackEpochCallback), partial(SaveModelCallback, every='epoch', name='model')]


print_model_mod(model)
learn.fit_one_cycle(3)
#learn.fit(500)
print(learn.recorder.lrs)
print(len(learn.recorder.lrs))
#learn.save('alex')
#learn.fit_one_cycle(35, 3e-3, wd=0.4)
