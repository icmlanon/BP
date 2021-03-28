Dependencies:
-	Pytroch	1.7.1
-	fastai		1.0.40

Example how to run on CIFAR10:
python cifar_train.py --model bp_resnet18 --gpu 0 --batch 64 --loss BP --r 1.0 --epochs 300

Bitlengths are printed as: [w1, a1, w2, a2, w3, a3...]

To print bitlenghts in each epoch:
python print_cifar.py --model bp_resnet18 --epochs 300

For ImageNet update the path of the dataset in lines #50:

python imagenet_train.py --model bp_resnet18 --gpu 0 --epochs 180 --batch 64 --loss BP --r 1.0 --lr 0.01

python print_imagenet.py --model bp_resnet18 --epochs 180



