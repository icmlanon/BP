"""Script to train imagenet model with preselected bitlenghts"""
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
    """Select learning parameters"""
    #tot_epochs,size,bs,lr =240,224,128,0.001
    #tot_epochs,size,bs,lr =35,224,64,0.001
    tot_epochs,size,bs,lr = 180,224,256,0.01
    #tot_epochs,size,bs,lr = 90,224,64,0.1
    dirname = 'ILSVRC2012_full'

    #gpu = setup_distrib(gpu)
    #if gpu is None: bs *= torch.cuda.device_count()
    torch.cuda.set_device(0)
    n_gpus = 1#num_distrib() or 1
    #n_gpus = num_distrib() or 1
    workers = min(12, num_cpus()//n_gpus)
    data = get_data(path/dirname, size, bs, workers)

    opt_func = partial(optim.SGD, momentum=0.9)
    
    """Select model"""
    #print_model_base(model_base)
    model=quant_alexnet()
    #model=models.alexnet()
    #model=quant_resnet18()
    #model=resnet18()
    #model=Quant_MobileNetV2()
	
	
    """Select loss function"""
    loss_func = nn.CrossEntropyLoss()#Quant_Loss(model, 0.01) 
    #loss_func = LabelSmoothingCrossEntropy()
	
	
    learn = Learner(data, model, metrics=[accuracy,top_k_accuracy], wd=0,
        opt_func=opt_func, bn_wd=False, true_wd=False,
        loss_func = loss_func).mixup(alpha=0.2)
    learn.callback_fns += [
        partial(TrackEpochCallback),
        partial(SaveModelCallback, every='epoch', name='fix')
    ]
    #copy_weights(model, model_base)





    #learn.load('/home/datasets/ILSVRC2012_full/models/org_fix_29')
    #learn.load('/home/datasets/ILSVRC2012_full/trained_models/alex_10')
    #learn.load('/home/datasets/ILSVRC2012_full/trained_models/res_10')
	"""Select bitlengths"""
    wgts=[4,4,4,4,4,4,4,4]
    acts=[4,4,4,4,4,4,4,4]
    i=0
    for m in learn.model.modules():
        if isinstance(m, Quant_Conv2d) or isinstance(m, Quant_IP):
            print(m.alpha_w.data,m.alpha_a.data)
            m.alpha_w.data=torch.FloatTensor([np.ceil(wgts[i])]).cuda()
            m.alpha_a.data=torch.FloatTensor([np.ceil(acts[i])]).cuda()
            m.alpha_w.trainable = False
            m.alpha_a.trainable = False
            m.alpha_w.requires_grad = False
            m.alpha_a.requires_grad = False
            i=i+1

    #learn.loss_func = nn.CrossEntropyLoss()


    print_model(learn.model)
    learn.split(lambda m: (children(m)[-2],))
    #if gpu is None: learn.model = nn.DataParallel(learn.model)
    #else:           learn.to_distributed(gpu)
    #learn.to_fp16(dynamic=True)
    torch.cuda.set_device(0)

    # Using bs 256 on single GPU as baseline, scale the LR linearly
    tot_bs = bs*n_gpus
    bs_rat = tot_bs/256
    lr *= bs_rat
	
	
    """Train"""
    learn.fit_one_cycle(tot_epochs, lr, div_factor=5, moms=(0.9,0.9),wd=0)
    learn.save('done_fin')
    print_model(model)

