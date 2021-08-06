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
from utils import load_samples
import datetime
os.environ['CUDA_VISIBLE_DEVICES']='0'

def inferen(args, split, modelfn=None, decoder=None, encoder=None,nllloss = None,centerloss=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dependency_model not in args.model_path:
        args.model_path += "_" + args.dependency_model
        if args.finetune_cnn:
            args.model_path += "_finetune"
    if encoder == None:
        modelfn = os.path.join(args.model_path, 'best_model.ckpt')
    img_name, img_tensor  = load_samples('samples-annotated')
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
    running_correct = 0.0
    count = 0
    running_loss = 0.0
    sample_num = 0.0
    init_time = datetime.datetime.now()
    for i in range(100):
        print(i)
        with torch.no_grad():
            images = img_tensor.to(device)
            features = encoder(images)
            ip1, outputs = decoder(features)
            _, pred = torch.max(outputs.data, 1)
    index = np.squeeze(np.argwhere(pred.cpu().numpy()==0))
    print(index)
    end_time = datetime.datetime.now()
    print((end_time-init_time).seconds*1000)
    for j in index:
        print(img_name[j])
    print('finished')

def main(args):
    inferen(args, "test")

if __name__ == "__main__":
    args = get_opt()
    main(args)
