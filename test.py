import argparse, os, time, pickle, json
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from models.image_models import ResNetFeats
from opts import get_opt
from optim import NoamOpt, LabelSmoothing
import models
from dataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from config import *

def test(args, split, modelfn=None, decoder=None, encoder=None,nllloss = None,centerloss=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dependency_model not in args.model_path:
        args.model_path += "_" + args.dependency_model
        if args.finetune_cnn:
            args.model_path += "_finetune"
    if encoder == None:
        modelfn = os.path.join(args.model_path, 'best_model.ckpt')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD), transforms.ToPILImage()])
    test_data = MyDataset(root=args.dir_name, datatxt=args.test_file_name, transform=transform,target_transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,pin_memory=True,num_workers=8)
    num_batches = len(test_loader)
    print(('[DEBUG] Running inference on %s with %d batches' % (split.upper(), num_batches)))
    if modelfn is not None:
        print(('[INFO] Loading checkpoint %s' % modelfn))
        encoder = ResNetFeats(args)
        decoder = models.setup(args)
        encoder.cuda()
        decoder.cuda()
        checkpoint = torch.load(modelfn)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    decoder.eval()
    pred_captions = []
    running_correct = 0.0
    count = 0
    running_loss = 0.0
    sample_num = 0.0
    total = 0
    correct = 0
    target_num = torch.zeros((1, args.class_num))
    predict_num = torch.zeros((1, args.class_num))
    acc_num = torch.zeros((1, args.class_num))
    with torch.no_grad():
        for i, current_batch in enumerate(tqdm(test_loader)):
            element_img, all_ids, ele_ids, label = current_batch
            sample_num = sample_num + current_batch[0].shape[0]
            targets = label
            images = element_img.to(device)
            targets = targets.to(device)
            images = images.to(device)
            features = encoder(images)
            ip1, outputs = decoder(features)
            _, pred = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += pred.eq(targets.data).cpu().sum()
            pre_mask = torch.zeros(outputs.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)
            loss = nllloss(outputs, targets) + args.center_weight*centerloss(targets, ip1)
            running_loss += loss.data[0]
            count = count + 1
            running_correct += torch.sum(pred == targets.data)
    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    accuracy = acc_num.sum(1) / target_num.sum(1)
    recall = (recall.numpy()[0] * 100).round(3)
    precision = (precision.numpy()[0] * 100).round(3)
    F1 = (F1.numpy()[0] * 100).round(3)
    accuracy = (accuracy.numpy()[0] * 100).round(3)
    print('recall', " ".join('%s' % id for id in recall))
    print('precision', " ".join('%s' % id for id in precision))
    print('F1', " ".join('%s' % id for id in F1))
    print('accuracy', accuracy)
    print("Test Loss is:{:.4f}, Test Accuracy is:{:.4f}%".format(running_loss / (sample_num), accuracy))
    encoder.train()
    decoder.train()
    return accuracy

def main(args):
    split = args.split
    test(args, split)

if __name__ == "__main__":
    args = get_opt()
    main(args)
