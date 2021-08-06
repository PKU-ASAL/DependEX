import numpy as np
import random
import torch
import os
import glob
from PIL import Image
from torchvision import transforms
from config import *

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_samples(img_dir):
    img_list = []
    img_glob = os.path.join(img_dir,'*.png')
    img_list.extend(glob.glob(img_glob))
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(MEAN, STD), transforms.ToPILImage()])
    target_transform = transforms.ToTensor()
    image_batch = []
    for img_name in img_list:
        element_img = Image.open(img_name).convert('RGB')
        preprocess_img = target_transform(transform(element_img).resize((224, 224), Image.ANTIALIAS))
        image_batch.append(preprocess_img.numpy())
    img_tensor_batch = torch.tensor(np.array(image_batch))
    return img_list,img_tensor_batch