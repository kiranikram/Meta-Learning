import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = './materials_aircrafts/'


class Aircrafts(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

"""## convnet encoding"""

from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

'''ResNet18 Encoding Variation'''
class ResNetEncoder(nn.Module):

    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.train()
        self.embedding = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        """
        Expects x to be of shape [batch_size, img_dim_1, img_dim_2]
        """
        #resize input to match the input dimension of ResNet
        resnet_input = nn.functional.interpolate(x, size=224)
        # here we can deal with the possibility of 1 input channel or 3
        if resnet_input.shape[1] == 1:
            resnet_input = resnet_input.repeat(1, 3, 1, 1)
        return self.embedding(resnet_input).view(x.size(0), -1)
    

''' Original paper setup'''
def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.Dropout(0.25),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

#model_ = ResNetEncoder()
#print(nn.Sequential(*list(model_.children(), nn.Dropout(0.5))[:-1]))

"""## Utils"""

import os
import shutil
import time
import pprint

import torch


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

"""## setting up the model and training"""

####### Define training setup #########

class Args:
    
    ''' use RESNET18 Embeddigs'''
    resnet_embedd = True
    
    '''main few shot learning parameters'''
    shot = 1
    query = 5
    train_way = 5
    test_way = 5
    
    ''' training epoch'''
    max_epoch = 200
    
    save_epoch = 10    
    save_path = f'./proto-{shot}s_{train_way}w_{query}q'
    gpu = 0
      
args=Args()

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


#from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#set_gpu(args.gpu)
ensure_path(args.save_path)

trainset = Aircrafts('train')
train_sampler = CategoriesSampler(trainset.label, 100,
                                  args.train_way, args.shot + args.query)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                          num_workers=8, pin_memory=True)

valset = Aircrafts('val')
val_sampler = CategoriesSampler(valset.label, 400,
                                args.test_way, args.shot + args.query)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                        num_workers=8, pin_memory=True)

if args.resnet_embedd==True:
    print('you are using ResNet embeddings!')
    model = ResNetEncoder().to(device)
else:
    model = Convnet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def save_model(name):
    torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))

trlog = {}
trlog['args'] = vars(args)
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = []
trlog['max_acc'] = 0.0

timer = Timer()
iteration = 0
for epoch in range(1, args.max_epoch + 1):


    model.train()

    tl = Averager()
    ta = Averager()

    for i, batch in enumerate(train_loader, 1):
        iteration += 1
        data, _ = [_.to(device) for _ in batch]
        p = args.shot * args.train_way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot)
        proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

        label = torch.arange(args.train_way).repeat(args.query)
        label = label.type(torch.LongTensor).to(device)

        logits = euclidean_metric(model(data_query), proto)
        loss = F.cross_entropy(logits, label)

        #writer.add_scalar('Loss/train', loss, iteration)

        acc = count_acc(logits, label)

        #writer.add_scalar('Acc/train', acc, iteration)

        print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
              .format(epoch, i, len(train_loader), loss.item(), acc))

        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        proto = None; logits = None; loss = None

    lr_scheduler.step()

    tl = tl.item()
    ta = ta.item()

    model.eval()

    vl = Averager()
    va = Averager()

    for i, batch in enumerate(val_loader, 1):
        data, _ = [_.to(device) for _ in batch]
        p = args.shot * args.test_way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot)
        proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

        label = torch.arange(args.test_way).repeat(args.query)
        label = label.type(torch.LongTensor).to(device)

        logits = euclidean_metric(model(data_query), proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        #writer.add_scalar('Loss/val', loss, iteration)
        #writer.add_scalar('Acc/val', acc, iteration)

        vl.add(loss.item())
        va.add(acc)

        proto = None; logits = None; loss = None

    vl = vl.item()
    va = va.item()
    print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    if va > trlog['max_acc']:
        trlog['max_acc'] = va
        save_model('max-acc')

    trlog['train_loss'].append(tl)
    trlog['train_acc'].append(ta)
    trlog['val_loss'].append(vl)
    trlog['val_acc'].append(va)

    torch.save(trlog, osp.join(args.save_path, 'trlog'))

    save_model('epoch-last')

    if epoch % args.save_epoch == 0:
        save_model('epoch-{}'.format(epoch))

    print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))



"""### Testing the model on  Test set"""

import argparse

import torch
from torch.utils.data import DataLoader


class Testing_Args:
    
    ''' use RESNET18 Embeddigs'''
    resnet_embedd = args.resnet_embedd
    
    '''main few shot learning parameters'''
    shot = 1
    query = 30
    way = 5
    
    batch = 100
    
    ''' load model'''
   
    load = './proto-1s_5w_5q/max-acc.pth'
    gpu = 0
      
test_args=Testing_Args()


dataset = Aircrafts('test')
sampler = CategoriesSampler(dataset.label,
                            test_args.batch, test_args.way, test_args.shot + test_args.query)
loader = DataLoader(dataset, batch_sampler=sampler,
                    num_workers=8, pin_memory=True)

if args.resnet_embedd==True:
    model = ResNetEncoder().to(device)
    model.load_state_dict(torch.load(test_args.load))
    model.eval()
else:    
    model = Convnet().to(device)
    model.load_state_dict(torch.load(test_args.load))
    model.eval()

ave_acc = Averager()

for i, batch in enumerate(loader, 1):
    data, _ = [_.to(device) for _ in batch]
    k = test_args.way * test_args.shot
    data_shot, data_query = data[:k], data[k:]

    x = model(data_shot)
    x = x.reshape(test_args.shot, test_args.way, -1).mean(dim=0)
    p = x

    logits = euclidean_metric(model(data_query), p)

    label = torch.arange(test_args.way).repeat(test_args.query)
    label = label.type(torch.LongTensor).to(device)

    acc = count_acc(logits, label)
    ave_acc.add(acc)
    print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

    x = None; p = None; logits = None

