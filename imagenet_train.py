"""Script to train bitlenghts for and imagenet model"""
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backends.cudnn.benchmark = True
import time
from quant_nets import *
from imagenet_quant_alexnet import *
from imagenet_quant_resnets import *
from imagenet_resnets import *
from imagenet_mobilenetv2 import *
from imagenet_resnext import *
from mnasnet import *
from imagenet_quant_mobilenetv2 import *
from mobilenet_v1 import *
from quant_mobilenet_v1 import *


import argparse

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


models_dict = {"resnet18": resnet18(num_classes=10), "mobilenetV2": mobilenet_v2(), "bp_alexnet": quant_alexnet(), "bp_resnet18": quant_resnet18(), "bp_mobilenetV2": Quant_MobileNetV2()}


parser = argparse.ArgumentParser(description='Train on ImageNet')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet18', choices=models_dict.keys(), help='model architecture: ' + ' | '.join(models_dict.keys()) + ' (default: resnet18)')
parser.add_argument('--gpu', type=int,default='0', help='gpu used for training (default: 0)')
parser.add_argument('--epochs', default=180, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--loss', default='BP', type=str, metavar='LOSS', help='loss function used (XE or BP)')
parser.add_argument('--r', '--regularizer', default=1.0, type=float, metavar='R', help='regularizer (default: 1.0)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='learning rate (default: 0.01)')
args = parser.parse_args()

"""Distributed training of Imagenet. Fastest speed is if you run with: python -m fastai.launch"""
path = Path('/home/datasets/')




"""Select learning parameters"""
tot_epochs,size,bs,lr = args.epochs, 224, args.batch, args.lr
dirname = 'ILSVRC2012_full'
torch.cuda.set_device(0)

"""Select model"""
model=models_dict[args.model]


"""Select loss function"""
if (args.loss=="XE"):
	loss_func = nn.LabelSmoothingCrossEntropy()
elif (args.loss=="BP"):
	loss_func = Quant_Loss(model=model,reg_strength=args.r)

n_gpus = 1
workers = min(12, num_cpus()//n_gpus)
data = get_data(path/dirname, size, bs, workers)

opt_func = partial(optim.SGD, momentum=0.9)



	

learn = Learner(data, model, metrics=[accuracy,top_k_accuracy], wd=1e-5, opt_func=opt_func, bn_wd=False, true_wd=False, loss_func = loss_func).mixup(alpha=0.2)
learn.callback_fns += [partial(SaveModelCallback, every='epoch', name='fix')]
learn.split(lambda m: (children(m)[-2],))
torch.cuda.set_device(args.gpu)
# Using bs 256 on single GPU as baseline, scale the LR linearly
tot_bs = bs*n_gpus
bs_rat = tot_bs/256
lr *= bs_rat


"""Train"""
learn.fit_one_cycle(tot_epochs, lr, div_factor=10, moms=(0.9,0.9))
learn.save('done')
print_model(model)

