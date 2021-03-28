Dependencies:
-	Pytroch
-	fastai

Select model, loss function and parameters within all scripts and then just run them with python script_name

Main file where all new functions are defined:
-	quant_nets.py

cifar scripts:
-	print_cifar.py				-Print bitlengths after each epoch
-	cifar_fin.py				-Script to fine-tune a cifar10 model after the integer bitlengths have been selected
-	cifar_print_dims.py			-Script for printing dimensions for a cifar10 model
-	cifar_train.py				-Script for training the bitlengths and weights for cifar10 models
-	cifar_train_wgt.py			-Script for targeted training of bitlengths and weights with a weighted loss function of cifar10 models
-	cifar_validate.py			-Script for validating a trained cifar model

cifar model definitions:
-	alex.py
-	quant_alexnet.py
-	quant_alexnet_bin.py
-	quant_alexnet_clip.py
-	quant_alexnet_pact.py
-	quant_densenet.py
-	quant_mobilenet_v1.py
-	quant_resnets.py
-	quant_shufflenetv2.py
-	cifar_quant_mobilenetv2.py
-	resnets.py
-	mnasnet.py
-	mobilenet_v1.py
-	models/*

imagenet scripts:
-	print_imagenet.py			-Print bitlengths after each epoch
-	imagenet_fin.py			-Script for finetuning with integer bitlengths for imagenet models
-	imagenet_load.py			-Script for printing imagenet model
-	imagenet_save.py			-Export imagenet model
-	imagenet_sbbkbb.py			-Script to train imagenet model with preselected bitlenghts
-	imagenet_train.py			-Script to train bitlenghts for and imagenet model
-	imagenet_train_wgt.py			-Script to train imagenet model and minimize any bitlength criteria
-	imagenet_validate.py			-Script for validating a trained imagenet model
-	imagenet_print_dims.py			-Script for printing dimensions for a imagenet model

imagenet model definitions:
-	imagenet_mobilenetv2.py
-	imagenet_quant_alexnet.py
-	imagenet_quant_mobilenetv2.py
-	imagenet_quant_resnets.py
-	imagenet_resnets.py
-	imagenet_resnext.py


