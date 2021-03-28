from fastai.script import *
from fastai.vision import *
from fastai.distributed import *
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import *

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair 

deterministic = True #False 

DEBUG= False #True
Alpha=8.0
Clip_Value=10.0



def isnan(x):
	return x != x

def print_model(model):
	for m in model.modules():
		if isinstance(m, Quant_Conv2d) or isinstance(m, Quant_IP):
			#print(m.weight.data[0],m.bias.data[0])
			print(m.alpha_w.data.cpu().numpy()[0],m.alpha_a.data.cpu().numpy()[0])
def print_model_dims(model):
	for m in model.modules():
		if isinstance(m, Quant_Conv2d) or isinstance(m, Quant_IP):
			print((m.weight.data.cpu().numpy()).size)
			#print((m.feature.cpu().numpy()).size)
def print_model_base(model):
	for m in model.modules():
		if isinstance(m, Conv2d) or isinstance(m, nn.Linear):
			print(m.weight.data[0],m.bias.data[0])
def print_model_weights(model):
	for m in model.modules():
		if isinstance(m, Quant_Conv2d) or isinstance(m, Quant_IP):
			print(m.weight.data,m.bias)

def print_model_mod(model):
	list=[]
	for m in model.modules():
		if isinstance(m, Quant_Conv2d) or isinstance(m, Quant_IP):
			list.append(m.alpha_w.data.cpu().numpy()[0])
			list.append(m.alpha_a.data.cpu().numpy()[0])
	print(list)

def copy_weights(model, model_base):
	weights=[]
	biases=[]
	for m in model_base.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			weights.append(m.weight.data)
			biases.append(m.bias.data)
	i=0
	for m in model.modules():
		if isinstance(m, Quant_Conv2d) or isinstance(m,Quant_IP):
			m.weight.data=weights[i]
			m.bias.data=biases[i]
			i=i+1
	print_model_weights(model)
	


class Quant_Loss(torch.nn.Module):
	def __init__(self,model,reg_strength):
		super(Quant_Loss,self).__init__()
		self.XEL_loss = nn.CrossEntropyLoss()
		self.Alpha_loss = nn.L1Loss()
		self.model=model
		self.reg_strength=reg_strength

	def forward(self,x,y):
		XEL = self.XEL_loss(x,y)
		alpha_reg=self.reg_strength#0.25
		max=0.0
		alpha=torch.tensor([0.0]).cuda()
		target=torch.tensor([0.0]).cuda()
		for m in self.model.modules():
			if isinstance(m, Quant_Conv2d) or isinstance(m, Quant_IP):# or isinstance(m, Quant_IP):
				if m.alpha_w.requires_grad:
					alpha+=m.alpha_w*alpha_reg
				if m.alpha_a.requires_grad:
					alpha+=m.alpha_a*alpha_reg
				max=max+2*8.0
		alpha_loss=self.Alpha_loss(alpha/max,target)
		total_loss = XEL+alpha_loss#*alpha_reg
		if DEBUG:
			print(alpha,total_loss,max)
		return total_loss 
		
		
class Quant_Loss_Weighted(torch.nn.Module):
	def __init__(self,model,wgts,acts,reg_strength):
		super(Quant_Loss_Weighted,self).__init__()
		self.XEL_loss = nn.CrossEntropyLoss()
		self.Alpha_loss = nn.L1Loss()
		self.model=model
		self.reg_strength=reg_strength
		self.wgts=wgts
		self.acts=acts

	def forward(self,x,y):
		XEL = self.XEL_loss(x,y)
		alpha_reg=1.0
		max=0.0
		alpha=torch.tensor([0.0]).cuda()
		target=torch.tensor([0.0]).cuda()
		i=0
		for m in self.model.modules():
			if isinstance(m, Quant_Conv2d) or isinstance(m, Quant_IP):# or isinstance(m, Quant_IP):
				alpha+=m.alpha_w*alpha_reg*self.wgts[i]
				alpha+=m.alpha_a*alpha_reg*self.acts[i]
				max=max+8.0*(self.wgts[i]+self.acts[i])
				i=i+1
		alpha_loss=self.Alpha_loss(alpha/max,target)
		#total_loss = XEL(1.0-alpha_reg)+weight_reg+torch.norm(alpha)*alpha_reg
		total_loss = XEL+alpha_loss#*alpha_reg
		if DEBUG:
			print(alpha,total_loss)
		return total_loss 
class Quant_Loss_0(torch.nn.Module):
	def __init__(self,model,reg_strength):
		super(Quant_Loss_0,self).__init__()
		self.XEL_loss = nn.CrossEntropyLoss()
		self.Alpha_loss = nn.L1Loss()
		self.model=model
		self.reg_strength=reg_strength

	def forward(self,x,y):
		XEL = self.XEL_loss(x,y)
		alpha_reg=0.0
		max=0.0
		alpha=torch.tensor([0.0]).cuda()
		target=torch.tensor([0.0]).cuda()
		for m in self.model.modules():
			if isinstance(m, Quant_Conv2d) or isinstance(m, Quant_IP):# or isinstance(m, Quant_IP):
				alpha+=m.alpha_w*alpha_reg
				alpha+=m.alpha_a*alpha_reg
				max=max+2*8.0
		alpha_loss=self.Alpha_loss(alpha/max,target)
		#total_loss = XEL(1.0-alpha_reg)+weight_reg+torch.norm(alpha)*alpha_reg
		total_loss = XEL+alpha_loss#*alpha_reg
		if DEBUG:
			print(alpha,total_loss)
		#weight_reg = (self.reg_strength/2)*torch.norm(weight) 
		return total_loss 



class RoundNoGradient(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		return x.round()
	@staticmethod
	def backward(ctx, g):
		return g




class Quantize_Interpolate(torch.autograd.Function):
	#@staticmethod
	def forward(x, alpha):
		low=torch.tensor(float(torch.floor(alpha)))#.cpu()
		high=low+1.0

		frac=alpha-low
		prob_h = (torch.rand(alpha.size()).cuda()<frac).float()
		prob_l = torch.abs(prob_h-1.0)



		quant_max = torch.max(x).cuda()
		quant_min = torch.min(x).cuda()
		quant_mq_h = torch.pow(torch.tensor(2.0),high).cuda() # number_of_steps 
		range_adjust_h = (quant_mq_h / (quant_mq_h - 1.0))
		quant_scale_h = range_adjust_h * (quant_max - quant_min) /quant_mq_h# range
		quant_int_h=RoundNoGradient.apply((x-quant_min)/quant_scale_h)


		quant_mq_l = torch.pow(torch.tensor(2.0),low).cuda() # number_of_steps 
		range_adjust_l = (quant_mq_l / (quant_mq_l - 1.0)) 
		quant_scale_l = range_adjust_l * (quant_max - quant_min) /quant_mq_l# range
		quant_int_l=RoundNoGradient.apply((x-quant_min)/quant_scale_l)
		quantized=(quant_min+quant_int_l*quant_scale_l*(high-alpha)+quant_int_h*quant_scale_h*(alpha-low)).cuda()
		return quantized.clone()
		
class Quant_IP(nn.Module):
	def __init__(self, input, out, init_value=1, interpolate=True, quantize=True, bits_w=8, bits_a=8,bias=True):
		super(Quant_IP,self).__init__()
		super(Quant_IP,self).__init__()
		self.in_features = input
		self.out_features = out
		self.weight = Parameter(torch.Tensor(out, input))
		if bias:
			self.bias = Parameter(torch.Tensor(out))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
		self.alpha_w = nn.Parameter(torch.FloatTensor([bits_w]).cuda(),requires_grad=True)
		self.alpha_a = nn.Parameter(torch.FloatTensor([bits_a]).cuda(),requires_grad=True)
		self.quantize=quantize
		self.bits_w=self.alpha_w
		self.bits_a=self.alpha_a

	def reset_parameters(self):
		init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)
	def forward(self, input,bits_w=8,bits_a=8):
                x=input
                if self.alpha_w<1.0:
                    #self.alpha_w.data=(torch.FloatTensor([1.0])).cuda()
                    self.alpha_w=nn.Parameter(torch.FloatTensor([1.0]).cuda(),requires_grad=False)

                if self.alpha_a<1.0:
                    #self.alpha_a.data=(torch.FloatTensor([1.0])).cuda()
                    self.alpha_a=nn.Parameter(torch.FloatTensor([1.0]).cuda(),requires_grad=False)
                if(self.quantize):
                    return F.linear(Quantize_Interpolate.forward(input,self.alpha_a), Quantize_Interpolate.forward(self.weight,self.alpha_w), self.bias)
                else:
                    return F.linear(input, self.weight, self.bias)
                return result
		
		
class Quant_IP_Individual(nn.Module):
	def __init__(self, input, out, init_value=1, interpolate=True, quantize=True, bits_w=8, bits_a=8,bias=True):
		super(Quant_IP_Individual,self).__init__()
		self.in_features = input
		self.out_features = out
		self.weight = Parameter(torch.Tensor(out, input))
		
		
		if bias:
			self.bias = Parameter(torch.Tensor(out))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
		self.alpha_w = nn.Parameter(torch.Tensor([[Alpha]*input]*out).cuda(),requires_grad=True)
		self.alpha_a = nn.Parameter(torch.Tensor([[Alpha]*input]*out).cuda(),requires_grad=True)
		self.quantize=quantize
		self.bits_w=self.alpha_w
		self.bits_a=self.alpha_a

	def reset_parameters(self):
		init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)
	def forward(self, input,bits_w=8,bits_a=8):
		self.bits_w=self.alpha_w
		self.bits_a=self.alpha_a
		x=input
		for x in self.alpha_w:
			for y in x:
				if y<1.0:
					y.data=(torch.FloatTensor([1.0])).cuda()
		for x in self.alpha_a:
			for y in x:
				if y<1.0:
					y.data=(torch.FloatTensor([1.0])).cuda()
		if(self.quantize):
			#return F.linear(input, self.weight, self.bias)
			return F.linear(Quantize_Interpolate_Individual.forward(input,self.alpha_a), Quantize_Interpolate_Individual.forward(self.weight,self.alpha_w), self.bias)
		else:
			return F.linear(input, self.weight, self.bias)
		return result
		
		
class _Quant_ConvNd(nn.Module):

	__constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

	def __init__(self, in_channels, out_channels, kernel_size, stride,
				 padding, dilation, transposed, output_padding,
				 groups, bias, padding_mode, interpolate,quantize,bits_w,bits_a):
		super(_Quant_ConvNd, self).__init__()
		if in_channels % groups != 0:
			raise ValueError('in_channels must be divisible by groups')
		if out_channels % groups != 0:
			raise ValueError('out_channels must be divisible by groups')
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.transposed = transposed
		self.output_padding = output_padding
		self.groups = groups
		self.padding_mode = padding_mode
		self.alpha_w = nn.Parameter(torch.FloatTensor([bits_w]).cuda(),requires_grad=True)
		self.alpha_a = nn.Parameter(torch.FloatTensor([bits_a]).cuda(),requires_grad=True)
		self.bits_w=self.alpha_w
		self.bits_a=self.alpha_a
		self.quantize=quantize
		if transposed:
			self.weight = Parameter(torch.Tensor(
				in_channels, out_channels // groups, *kernel_size))
		else:
			self.weight = Parameter(torch.Tensor(
				out_channels, in_channels // groups, *kernel_size))
		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		n = self.in_channels
		init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def extra_repr(self):
		s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
			 ', stride={stride}')
		if self.padding != (0,) * len(self.padding):
			s += ', padding={padding}'
		if self.dilation != (1,) * len(self.dilation):
			s += ', dilation={dilation}'
		if self.output_padding != (0,) * len(self.output_padding):
			s += ', output_padding={output_padding}'
		if self.groups != 1:
			s += ', groups={groups}'
		if self.bias is None:
			s += ', bias=False'
		return s.format(**self.__dict__)


class Quant_Conv1d(_Quant_ConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1,
				 bias=True, padding_mode='zeros'):
		kernel_size = _single(kernel_size)
		stride = _single(stride)
		padding = _single(padding)
		dilation = _single(dilation)
		super(Conv1d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			False, _single(0), groups, bias, padding_mode)

	def forward(self, input):
		if isnan(self.alpha_w):
			self.alpha_w=self.bits_w
			print("Alpha W is NAN")
		if isnan(self.alpha_a):
			self.alpha_a=self.bits_a
			print("Alpha A is NAN")
		self.bits_w=self.alpha_w
		self.bits_a=self.alpha_a
		if self.alpha_w<1.0:
			self.alpha_w.data=(torch.FloatTensor([1.0])).cuda()
			self.alpha_w.trainable = False
			self.alpha_w.requires_grad = False
			print("Alpha W below 1")
		if self.alpha_a<1.0:
			self.alpha_a.data=(torch.FloatTensor([1.0])).cuda()
			self.alpha_a.trainable = False
			self.alpha_a.requires_grad = False
			print("Alpha A below 1")
		if self.padding_mode == 'circular':
			expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
			return F.conv1d(F.pad(input, expanded_padding, mode='circular'), self.weight, self.bias, self.stride, _single(0), self.dilation, self.groups)
		if self.quantize:
			return F.conv1d(Quantize_Interpolate.forward(input,self.alpha_a), Quantize_Interpolate.forward(self.weight,self.alpha_w), self.bias, self.stride, self.padding, self.dilation, self.groups)
		else:
			return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class Quant_Conv2d(_Quant_ConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', interpolate=True,quantize=True,bits_w=8,bits_a=8):
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		super(Quant_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode,interpolate,quantize,bits_w,bits_a)

	def forward(self, input):
            if self.alpha_w<1.0:
                self.alpha_w=nn.Parameter(torch.FloatTensor([1.0]).cuda(),requires_grad=False)
            if self.alpha_a<1.0:
                self.alpha_a=nn.Parameter(torch.FloatTensor([1.0]).cuda(),requires_grad=False)
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2, (self.padding[0] + 1) // 2, self.padding[0] // 2)
                return F.conv2d(F.pad(Quantize_Interpolate.forward(input,self.alpha_a), expanded_padding, mode='circular'), Quantize_Interpolate.apply(self.weight,self.alpha_w), self.bias, self.stride, _pair(0), self.dilation, self.groups)
            if self.quantize:
                return F.conv1d(Quantize_Interpolate.forward(input,self.alpha_a), Quantize_Interpolate.forward(self.weight,self.alpha_w), self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Quant_Conv3d(_Quant_ConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1,
				 bias=True, padding_mode='zeros'):
		kernel_size = _triple(kernel_size)
		stride = _triple(stride)
		padding = _triple(padding)
		dilation = _triple(dilation)
		super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _triple(0), groups, bias, padding_mode)

	def forward(self, input):
		if isnan(self.alpha_w):
			self.alpha_w=self.bits_w
			print("Alpha W is NAN")
		if isnan(self.alpha_a):
			self.alpha_a=self.bits_a
			print("Alpha A is NAN")
		self.bits_w=self.alpha_w
		self.bits_a=self.alpha_a
		if self.alpha_w<1.0:
			self.alpha_w.data=(torch.FloatTensor([1.0])).cuda()
			self.alpha_w.trainable = False
			self.alpha_w.requires_grad = False
			print("Alpha W below 1")
		if self.alpha_a<1.0:
			self.alpha_a.data=(torch.FloatTensor([1.0])).cuda()
			self.alpha_a.trainable = False
			self.alpha_a.requires_grad = False
			print("Alpha A below 1")
		
		if self.padding_mode == 'circular':
			expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2, (self.padding[1] + 1) // 2, self.padding[1] // 2, (self.padding[0] + 1) // 2, self.padding[0] // 2)
			return F.conv3d(F.pad(input, expanded_padding, mode='circular'), Quantize_Interpolate.apply(self.weight,self.bits_w,self.alpha_w), self.bias, self.stride, _triple(0), self.dilation, self.groups)
		if self.quantize:
			return F.conv3d(Quantize_Interpolate.forward(input,self.alpha_a), Quantize_Interpolate.apply(self.weight,self.alpha_w), self.bias, self.stride, self.padding, self.dilation, self.groups)
		else:
			return F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


