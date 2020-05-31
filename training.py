"""## Setting up Background (training) and Evaluation (test) dataloader"""

from torch.utils.data import DataLoader
from torch import nn
import argparse
from torchvision import transforms

import PIL


setup_dirs()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = 'aircrafts'
dataset_class = Aircraft
fc_layer_size = 1600
num_input_channels = 3


param_str = f'{dataset}_order={order}_n={n}_k={k}_metabatch={meta_batch_size}_' \
            f'train_steps={inner_train_steps}_val_steps={inner_val_steps}'
print(param_str)


###################
# Create datasets #
###################

if a>0:
    meta_batch_size=1
epoch_len = int(epoch_len*32/meta_batch_size)      

background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, epoch_len, n=n, k=k, q=q,
                                   num_tasks=meta_batch_size),
    num_workers=8
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, eval_batches, n=n, k=k, q=q,
                                   num_tasks=meta_batch_size),
    num_workers=8
)

print('iterations per epoch: ',epoch_len)

"""## Examples of Data Augmentation"""

class Augment__():
    def __init__(self, rotation=0, hue=0.5, saturation=0):
        
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
            
        return transforms_(z)
    
    def no_aug_(self, x):
        
        z = x * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        z = z + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        
        transforms_ = transforms.Compose([
            transforms.ToPILImage(),
            
        ])

            
        return transforms_(z)
    
example_data = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, eval_batches, n=n, k=k, q=q,
                                   num_tasks=1),
    num_workers=8
)
    
import matplotlib.pyplot as plt

aug_examples = Augment__(rotation=10, hue=0.2, saturation=3)

nr_imgs = 4

fig, axs = plt.subplots(2, nr_imgs, figsize=(20,10))


t = 0
augs = 0
#plt.subplot(5,2)
for b in example_data:
    
    t_img = b[0][0]
    
    img_s = aug_examples.no_aug_(b[0][0])
    aug_s = aug_examples.aug_(t_img)
    #print(t)
    axs[0, t].imshow(img_s)
    axs[0, t].set_title(['Background Images'])
    axs[1, t].imshow(aug_s)
    axs[1, t].set_title(['Augmented Images'])
    
    t+=1
    augs = t+1
    if t==nr_imgs:
        break
plt.subplots_adjust(hspace=0.3)    
plt.show()

"""## MAML Training"""

############
# Training #
############


print(f'Training MAML on {dataset}...')
meta_model = FewShotClassifier(num_input_channels, k, fc_layer_size).to(device, dtype=torch.double)
meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
loss_fn = nn.CrossEntropyLoss().to(device)

'''preparing batches for training'''
def prepare_meta_batch(n, a, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        
        x = x.reshape(meta_batch_size, n*(1+a)*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
       
        #print('x.reshape---> ', x.shape)
        # Move to device
        x = x.double().to(device)
        # Create label
        y = create_nshot_task_label(k, q).to(device).repeat(meta_batch_size)
        
        return x, y

    return prepare_meta_batch_

'''preparing batches for evaluation'''
def prepare_meta_batch_eval(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        
        x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
       
        #print('x.reshape---> ', x.shape)
        # Move to device
        x = x.double().to(device)
        # Create label
        y = create_nshot_task_label(k, q).to(device).repeat(meta_batch_size)
        
        return x, y

    return prepare_meta_batch_



callbacks = [
    EvaluateFewShot(
        eval_fn=meta_gradient_step_eval,
        num_tasks=eval_batches,
        n_shot=n,
        k_way=k,
        q_queries=q,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch_eval(n, k, q, meta_batch_size),
        # MAML kwargs
        inner_train_steps=inner_val_steps,
        inner_lr=inner_lr,
        device=device,
        order=order,
    ),
    ModelCheckpoint(
        #filepath=PATH + f'/models/maml_aircraft/{param_str}.pth',
        filepath=f'./models/maml_aircraft/{param_str}.pth',
        monitor=f'val_{n}-shot_{k}-way_acc'
    ),
    ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
    #CSVLogger(PATH + f'/logs/maml_aircraft/{param_str}.csv'),
    CSVLogger(f'./logs/maml_aircraft/{param_str}.csv'),
]


fit(
    meta_model,
    meta_optimiser,
    loss_fn,
    epochs=epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_meta_batch(n, a, k, q, meta_batch_size),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=meta_gradient_step,
    fit_function_kwargs={'n_shot': n,'a_aug': a, 'k_way': k, 'q_queries': q, 
                         #'meta_batch_size': meta_batch_size,
                         'train': True,
                         'order': order, 'device': device, 'inner_train_steps': inner_train_steps,
                         'inner_lr': inner_lr},
)

