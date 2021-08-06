import argparse, os, time, pickle, random, sys
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from tensorboardX import SummaryWriter
from models.image_models import ResNetFeats
from opts import get_opt
from optim import NoamOpt
import models
from CenterLoss import CenterLoss
from dataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from test import test
from config import *
from utils import set_seed
os.environ['CUDA_VISIBLE_DEVICES']='3'
torch.backends.cudnn.enabled = False

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	args.model_path += "_" + args.dependency_model
	if args.finetune_cnn:
		args.model_path += "_finetune"
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	writer = SummaryWriter(log_dir=args.model_path)
	transform = transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),transforms.ToTensor(), transforms.Normalize(MEAN, STD), transforms.ToPILImage()])
	train_data = MyDataset(root=args.dir_name, datatxt=args.train_file_name, transform=transform,target_transform=transforms.ToTensor())
	class_sample_counts = [4642,6802,8668]
	class_weights = 1./torch.Tensor(class_sample_counts)
	train_targets = [sample[4] for sample in train_data.imgs]
	train_samples_weight = [class_weights[class_id] for class_id in train_targets]
	train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, len(train_data))
	train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, sampler=train_sampler,pin_memory=True,num_workers=8)
	print("# args.img_fatures_size:", args.img_fatures_size)
	encoder = ResNetFeats(args)
	decoder = models.setup(args)
	encoder.to(device)
	encoder.train(True)
	decoder.to(device)
	decoder.train(True)
	nllloss = nn.NLLLoss().to(device)
	centerloss = CenterLoss(args.class_num, 30).to(device)
	params = list(decoder.parameters())
	if args.finetune_cnn:
		params += list(encoder.resnetLayer4.parameters())
	params += list(encoder.adaptive_pool7x7.parameters())
	optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.98), eps=1e-9)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
	start_epoch = 0
	loss_train = 0
	total_step = len(train_loader)
	iteration = start_epoch * total_step
	bestiter = iteration
	train_start = time.time()
	best_acc = 0
	best_epoch = 0
	for epoch in range(start_epoch, args.num_epochs):
		print("\n==>Epoch:", epoch)
		running_loss = 0.0
		running_correct = 0.0
		count = 0
		sample_num = 0.0
		for i, current_batch in enumerate(tqdm(train_loader)):
			element_img, all_ids, ele_ids, label = current_batch
			sample_num = sample_num + current_batch[0].shape[0]
			targets = label
			images = element_img.to(device)
			targets = targets.to(device)
			features = encoder(images)
			ip1, outputs = decoder(features)
			_, pred = torch.max(outputs.data, 1)
			loss = nllloss(outputs, targets) + args.center_weight * centerloss(targets, ip1)
			writer.add_scalar("Loss/train", loss, iteration)
			decoder.zero_grad()
			encoder.zero_grad()
			loss.backward()
			optimizer.step()
			iteration += 1
			count += 1
			running_loss += loss.data[0]
			running_correct += torch.sum(pred == targets.data)
		scheduler.step()
		print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%".format(running_loss / (sample_num),100 * running_correct.float() / (sample_num)))
		test_acc = test(args, "test", encoder=encoder, decoder=decoder,nllloss = nllloss,centerloss=centerloss)
		if test_acc> best_acc:
			best_acc = test_acc
			best_epoch = epoch
			print("[INFO] save model")
			save_path = os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1))
			optim_state_dict = optimizer.state_dict()
			torch.save({'epoch': epoch,'decoder_state_dict': decoder.state_dict(),'encoder_state_dict': encoder.state_dict(),'optimizer': optim_state_dict,'iteration': iteration,'accuracy': test_acc,}, save_path)
			bestiter = iteration
			print(('[DEBUG] Saving model at epoch %d with accuracy of %f' % (epoch, test_acc)))
			bestmodel_path = os.path.join(args.model_path, 'best_model.ckpt')
			os.system('cp %s %s' % (save_path, bestmodel_path))
	test_acc = test(args, "test", encoder=encoder, decoder=decoder,nllloss = nllloss,centerloss=centerloss)
	if test_acc > best_acc:
		best_acc = test_acc
		best_epoch = epoch
		print("[INFO] save model")
		save_path = os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1))
		optim_state_dict = optimizer.state_dict()
		torch.save({'epoch': epoch,'decoder_state_dict': decoder.state_dict(),'encoder_state_dict': encoder.state_dict(),'optimizer': optim_state_dict,'iteration': iteration,'accuracy': test_acc,}, save_path)
		bestiter = iteration
		print(('[DEBUG] Saving model at epoch %d with accuracy of %f' % (epoch, test_acc)))
		bestmodel_path = os.path.join(args.model_path, 'best_model.ckpt')
		os.system('cp %s %s' % (save_path, bestmodel_path))
	print('finished')
	print(best_epoch,best_acc)

if __name__ == '__main__':
	set_seed(1234)
	args = get_opt()
	print(args)
	main(args)
