#The code below for the MAML model implementation has been adapted by the github repository: https://github.com/oscarknagg/few-shot

#Detailed are below is the DATA AUGMANTATION enhancement to the original implementation: the hyperparameters for the data augmentation can be modified in the cell called: META Learning Parameters selection


from torch.utils.data import DataLoader
from torch import nn
import argparse
from torchvision import transforms
import PIL

"""##### Temporary parameters selection: they can be modified below, before the actual training"""

##### PARAMETERS #######

n = 1 # Number of samples for each class
k = 5 # Number of classes in the n-shot classification tasks
q = 5 # Number query samples for each class in the n-shot classification tasks

inner_train_steps = 5
inner_val_steps = 5
inner_lr = 0.01
meta_lr = 0.001
meta_batch_size = 5  #32
order = 1
epochs = 2
epoch_len = 10 
eval_batches = 40

#### AUGMENTATION PARAMETERS #####

a = 0 # Number of samples augmented for each class

rotation=0 
hue=0 
saturation=0

"""### Prepare Data"""

from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


PATH = '.' #os.path.dirname(os.path.realpath(__file__))

DATA_PATH = './DATA_PATH'

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')

class Aircraft(Dataset):
    def __init__(self, subset):
    
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dictionaries
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(800),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        #print(item)
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):

        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/aircrafts/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/aircrafts/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images

"""## Creating directories for CSV Log Files"""

#These are the directories where the results will be stored
import shutil
from typing import Tuple, List

def mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        pass

def rmdir(dir):
    try:
        shutil.rmtree(dir)
    except:
        pass

#make sure the diectories do not already exist when running it fresh
def setup_dirs():
    """Creates directories for this project."""
    mkdir(PATH + '/logs/')
    mkdir(PATH + '/logs/maml')
    mkdir(PATH + '/models/')
    mkdir(PATH + '/models/maml')

"""## Here we randomly split the 100 classes into 80 in background and 20 in evaluation"""

from tqdm import tqdm as tqdm
import numpy as np
import shutil
import os

# Clean up folders
rmdir(DATA_PATH + '/aircrafts/images_background')
rmdir(DATA_PATH + '/aircrafts/images_evaluation')
mkdir(DATA_PATH + '/aircrafts/images_background')
mkdir(DATA_PATH + '/aircrafts/images_evaluation')

# Find class identities
#Classes in Background and Evaluation should not be the same 
classes = []
for root, _, files in os.walk(DATA_PATH + '/aircrafts/images/'):
    for f in files:
        if f.endswith('.jpg'):
            classes.append(f[:-12])

classes = list(set(classes))

# Train/test split within each of background and evaluation 
np.random.seed(0)
np.random.shuffle(classes)
background_classes, evaluation_classes = classes[:80], classes[80:]

# Create class folders
for c in background_classes:
    mkdir(DATA_PATH + f'/aircrafts/images_background/{c}/')

for c in evaluation_classes:
    mkdir(DATA_PATH + f'/aircrafts/images_evaluation/{c}/')

# Move images to correct location
for root, _, files in os.walk(DATA_PATH + '/aircrafts/images'):
    for f in tqdm(files, total=100*100):
        if f.endswith('.jpg'):
            class_name = f[:-12]
            image_name = f[-12:]
            # Send to correct folder
            subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
            src = f'{root}/{f}'
            dst = DATA_PATH + f'/aircrafts/{subset_folder}/{class_name}/{image_name}'
            shutil.copy(src, dst)

"""## Accuracy calculated when META TRAINING (ie learning on multiple tasks)"""

def categorical_accuracy(y, y_pred):

    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]
NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy
}