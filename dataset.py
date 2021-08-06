from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(root + datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.strip()
            items = line.split('|')
            if items[4] == 'ia':
                label = 0
            elif items[4] == 'li':
                label = 1
            else:
                label = 2
            imgs.append((items[0], items[1], items[2], items[3], label))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        raw_img_path, element_img_path, all_ids, ele_ids, label = self.imgs[index]
        element_img = Image.open(element_img_path).convert('RGB')
        if self.transform is not None:
            element_img = self.target_transform(self.transform(element_img).resize((224, 224), Image.ANTIALIAS))
        return element_img, all_ids, ele_ids, label

    def __len__(self):
        return len(self.imgs)

def data_batch():
    dir_name = 'E:\\OSLAB\\LearnDependency\\DATA\\'
    file_name = 'label_input_action_truth.txt'
    train_data = MyDataset(root=dir_name, datatxt=file_name, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    use_cuda = True
    for batch in train_loader:
        raw_img,element_img,all_ids, ele_ids, label = batch
        if use_cuda:
            data, target = raw_img.cuda(), element_img.cuda()
    print('finished')

def data_generate():
    dir_name = 'E:\\OSLAB\\LearnDependency\\data_bank\\'
    file_name = 'train.txt'
    PIXEL_MEANS = (0.485, 0.456, 0.406)  # RGB  format mean and variances
    PIXEL_STDS = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(PIXEL_MEANS, PIXEL_STDS),transforms.ToPILImage()])
    train_data = MyDataset(root=dir_name, datatxt=file_name, transform=transform,target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    use_cuda = True
    for batch in train_loader:
        raw_img, element_img, all_ids, ele_ids, label = batch
        if use_cuda:
            data, target = raw_img.cuda(), element_img.cuda()
    print('finished')

if __name__=='__main__':
    data_generate()
