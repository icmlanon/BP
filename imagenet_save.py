"""Export imagenet model"""
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backends.cudnn.benchmark = True
import time
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


from quant_nets import *
from imagenet_quant_alexnet import *
from imagenet_quant_resnets import *
from imagenet_resnets import *
from imagenet_quant_mobilenetv2 import *

def get_data(path, size, bs, workers):
    tfms = ([
        flip_lr(p=0.5),
        brightness(change=(0.4,0.6)),
        contrast(scale=(0.7,1.3))
    ], [])
    train = ImageList.from_csv(path/'train', 'train.csv')#.use_partial_data(0.001)
    valid = ImageList.from_csv(path/'val_org', 'valid.csv')#.use_partial_data()
    lls = ItemLists(path, train, valid).label_from_df().transform(
            tfms, size=size).presize(size, scale=(0.35, 1.0))
    return lls.databunch(bs=bs, num_workers=workers).normalize(imagenet_stats)

@call_parse
def main( gpu:Param("GPU to run on", str)=None ):
    """Distributed training of Imagenet. Fastest speed is if you run with: python -m fastai.launch"""
    path = Path('/home/datasets/')
    tot_epochs,size,bs,lr = 60,224,128,0.001
    #tot_epochs,size,bs,lr = 90,224,64,0.1
    dirname = 'ILSVRC2012_full'

    torch.cuda.set_device(0)
    n_gpus = 1#num_distrib() or 1
    workers = min(12, num_cpus()//n_gpus)
    data = get_data(path/dirname, size, bs, workers)

    opt_func = partial(optim.SGD, momentum=0.9)
	
	
	"""Select model architecture"""
    #model=quant_alexnet()
    model=quant_resnet18()
    loss_func = nn.CrossEntropyLoss()#Quant_Loss(model, 0.01) 
    learn = Learner(data, model, metrics=[accuracy,top_k_accuracy], wd=0,
        opt_func=opt_func, bn_wd=False, true_wd=False,
        loss_func = loss_func).mixup(alpha=0.2)
    learn.callback_fns += [
        partial(TrackEpochCallback),
        partial(SaveModelCallback, every='epoch', name='fix')
    ]






    """Load trained model"""
    #learn.load('/home/datasets/ILSVRC2012_full/trained_models/alex_10_fin')
    learn.load('/home/datasets/ILSVRC2012_full/trained_models/res_10_fin_90')

    torch.save(learn.model.state_dict(), "./res_trained_10.pth")
