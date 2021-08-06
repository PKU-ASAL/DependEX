import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import copy


class ResNetFeats(nn.Module):
	def __init__(self, args):
		super(ResNetFeats, self).__init__()
		self.dependency_model = args.dependency_model
		self.att_size = getattr(args, "att_size", 7)
		self.embed_size = getattr(args, "embed_size", 4096)
		self.finetune_cnn = getattr(args, "finetune_cnn", False)
		resnet = models.resnet101(pretrained=True)
		if not self.finetune_cnn:
			modules = list(resnet.children())[:-2]
		else:
			modules = list(resnet.children())[:-3] 
			self.resnetLayer4 = resnet.layer4
		self.resnet = nn.Sequential(*modules)
		self.adaptive_pool1x1 = list(resnet.children())[-2]
		self.adaptive_pool7x7 = nn.AdaptiveAvgPool2d((self.att_size, self.att_size))
		self.linear = nn.Linear(resnet.fc.in_features, self.embed_size)
		self.bn = nn.BatchNorm1d(self.embed_size, momentum=0.01)

	def forward(self, images):
		with torch.no_grad():
			x = self.resnet(images)
		if self.finetune_cnn:
			x = self.resnetLayer4(x)
		if self.dependency_model == "transformer":
			att = self.adaptive_pool7x7(x).squeeze().permute(0, 2, 3, 1)
			att = att.view(images.size(0), -1, att.size(-1))
			return att
				

