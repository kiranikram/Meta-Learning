"""## Training and Meta Training

The fit function allows for meta training parameters to be called
"""

import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union
from torchvision import transforms
#from callbacks import DefaultCallback, ProgressBarLogger, CallbackList, Callback
#from metrics import NAMED_METRICS
import PIL


class Augment():
    def __init__(self, rotation=rotation, hue=hue, saturation=saturation):
        
        self.rotation = rotation
        self.hue = hue
        self.saturation = saturation

    def aug_(self, x):
        
        z = x * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        z = z + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        
        transforms_ = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(self.hue, self.saturation),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(self.rotation, resample=PIL.Image.BILINEAR),
            # transforms.CenterCrop(20),
            #transforms.ToTensor()
        ])

        transform_toT = transforms.Compose([
            transforms.Resize(84),
            #transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
            
        return transform_toT(transforms_(z))

aug = Augment()

''' BATCH modification for DATA Augmentation'''
def modify_batch(batch, n, k, a, meta_batch_size):
    x, y = batch

    data_shot_aug = x[:n*k*meta_batch_size]
    qx = x[n*k*meta_batch_size:]
    #print('qx.shape',qx.shape)
    data_shot_aug_y = y[:n*k*meta_batch_size]
    y_q = y[n*k*meta_batch_size:]
    
    augs_y = []
    for s in range(data_shot_aug.shape[0]):
        #print('s', s)
        augs = data_shot_aug[s]
        
        augs_ = augs.unsqueeze(0)
        augs_y.append(data_shot_aug_y[s])
        
        for a_ in range(a):
            #print('a', a_)
            new = aug.aug_(augs)
            #print('new',new.shape)
            
            augs_ = torch.cat([augs_, new.unsqueeze(0)])
            augs_y.append(data_shot_aug_y[s])
            #print(new)
        data_shot_aug = torch.cat([data_shot_aug,augs_])
        #data_shot_aug_y = torch.cat([data_shot_aug_y.reshape(-1,),augs_y])
        
    data_shot_aug = data_shot_aug[n*k*meta_batch_size:]
    augs_y = torch.stack(augs_y)
    #print('data_shot_aug.shape',data_shot_aug.shape)
    #print('augs_y',augs_y)
   
    x = torch.cat([data_shot_aug, qx])
    y = torch.cat([augs_y, y_q])
    #print('xxx.shape',x.shape)
    #print('(n + 1)*k + q*k', (n + 1)*k + q*k)  
    #x = x.reshape(meta_batch_size, (n*(1+a))*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
    #print(x.shape)
    
    batch = (x, y)
    
    return batch





def gradient_step(model: Module, optimiser: Optimizer, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs):

    model.train()
    optimiser.zero_grad()
    #print('x.shape in fit_func', x.shape)
    y_pred = model(x)
    #print(y_pred)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred


def batch_metrics(model: Module, y_pred: torch.Tensor, y: torch.Tensor, metrics: List[Union[str, Callable]],
                  batch_logs: dict):

    model.eval()
    for m in metrics:
        if isinstance(m, str):
            batch_logs[m] = NAMED_METRICS[m](y, y_pred)
        else:
            # Assume metric is a callable function
            batch_logs = m(y, y_pred)

    return batch_logs


def fit(model: Module, optimiser: Optimizer, loss_fn: Callable, epochs: int, dataloader: DataLoader,
        prepare_batch: Callable, metrics: List[Union[str, Callable]] = None, callbacks: List[Callback] = None,
        verbose: bool =True, fit_function: Callable = gradient_step, fit_function_kwargs: dict = {}):

    # Determine number of samples:
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': loss_fn,
        'optimiser': optimiser
    })

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()
    loss_ls = []
    for epoch in range(1, epochs+1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        loss_ls = []
        batch_logs_ = []
        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))
            #print('batch[0].shape',batch[0].shape)
            #print('batch[1].shape',batch[1].shape)
            callbacks.on_batch_begin(batch_index, batch_logs)
            #print('batch_index',batch_index)
            print('support + query samples *** before Augment ***', batch[0].shape)
            #******* data augmentation **************
            
            batch = modify_batch(batch, n=n, k=k, a=a, meta_batch_size=meta_batch_size) 
            print('a', a)
            print('support + query samples *** after Augment ***', batch[0].shape)
            
            
            x, y = prepare_batch(batch)
          

            loss, y_pred = fit_function(model, optimiser, loss_fn, x, y, **fit_function_kwargs)
            
            loss_ls.append(loss)
            
            batch_logs['loss'] = loss.item()

            # Loops through all metrics
            batch_logs = batch_metrics(model, y_pred, y, metrics, batch_logs)
            batch_logs_.append(batch_logs)
            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
        
        callbacks.on_epoch_end(epoch, epoch_logs)
        
    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()
    
    return loss_ls,batch_logs_