from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
import numpy as np
import torch
from torchvision import transforms
class Augment():
    def __init__(self, rotation=rotation, hue=hue, saturation=saturation):
        self.rotation = rotation
        self.hue = hue
        self.saturation = saturation

    def aug_(self, x):
        transforms_ = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(self.hue, self.saturation),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(self.rotation, resample=PIL.Image.BILINEAR),
            # transforms.CenterCrop(20),
            transforms.ToTensor()
        ])

        transform_toT = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return transform_toT(transforms_(x))


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):

        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n, replace=True)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    #print('self.q', self.q)
                    #print('df[(df[...shape',df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].shape)
                    if df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].shape[0]==0:
                        query = []
                    else:
                        #print('its OK!!!!!!!!')
                        query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                        for i, q in query.iterrows():
                            batch.append(q['id'])

            yield np.stack(batch)
