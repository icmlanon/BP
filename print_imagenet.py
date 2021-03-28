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
parser.add_argument('--epochs', default=180, type=int, metavar='N', help='number of total epochs to run')
args = parser.parse_args()

path = Path('/home/datasets/')
tot_epochs,size,bs,lr = 180,224,1,0.01
dirname = 'ILSVRC2012_full'

#gpu = setup_distrib(gpu)
#if gpu is None: bs *= torch.cuda.device_count()

#n_gpus = num_distrib() or 1
#workers = min(12, num_cpus()//n_gpus)

path = Path('/home/datasets/')
data = get_data(path/dirname, size, bs, 1)#workers)

opt_func = partial(optim.SGD, momentum=0.9)


"""Select model"""
model=models_dict[args.model]
model=quant_alexnet()
loss_func = LabelSmoothingCrossEntropy()
learn = Learner(data, model, metrics=[accuracy,top_k_accuracy], wd=1e-5, opt_func=opt_func, bn_wd=False, true_wd=False, loss_func = loss_func).mixup(alpha=0.2)


"""Load trained weights"""
for i in range(1, args.epochs):
	learn.load('fix_'+str(i))
	print_model_mod(learn.model)
