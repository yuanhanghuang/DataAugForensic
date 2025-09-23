import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T
import torch.multiprocessing

class ImageFolder(data.Dataset):
    def __init__(self, root, image_size, mode, is_distributed):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.image_size=image_size
        self.image_names = os.listdir(self.root)
        self.mode = mode

        if is_distributed:
            if torch.distributed.get_rank()==0:
                print("image count in {} path :{}".format(self.mode, len(self.image_names)))
        else:
            print("image count in {} path :{}".format(self.mode, len(self.image_names)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_name = self.image_names[index]
        image_path = os.path.join(self.root,image_name)
        image = Image.open(image_path).convert('RGB')

        GT = np.zeros((self.image_size,self.image_size), dtype=np.uint8)
        min_GT = GT.resize(size=(48,48),resample=Image.LANCZOS) if self.mode != 'launch' else None
        filename = None if self.mode != 'launch' else image_path.split('/')[-1]
        return image, GT, min_GT, filename

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_names)

def imag_trans(image):
    image_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.413, 0.406, 0.369], [0.269, 0.262, 0.283])
    ])
    
    return image_transforms(image)

def to_tensor(GT):
    GT[GT < 127.5] = 0
    GT[GT >= 127.5] = 255
    if len(GT.shape) >= 3:
        GT = GT[:, :, 0]

    GT = np.expand_dims(GT, axis=-1)
    GT = torch.from_numpy(GT / 255.).permute(2, 0, 1).float()

    return GT

def collate_fn_launch(batch):
    Transform = []
    Transform = T.Compose(Transform)

    images = []
    GTs = []
    filenames = []
    for value in batch:
        image = value[0]
        GT = value[1]
        filename = value[3]

        image = Transform(image)
        image = np.array(image)
        image = imag_trans(image)

        GT = Transform(GT)
        GT = np.array(GT)
        GT = to_tensor(GT)

        images.append(image)
        GTs.append(GT)
        filenames.append(filename)

    images = torch.stack(images, dim=0)
    GTs = torch.stack(GTs, dim=0)
    filenames = np.stack(filenames, axis=0)
    return images, GTs, filenames


def get_loader(image_path, mode, batch_size,image_size,num_workers, work_init_fn=None, generator=None, is_distributed=True):
    global size
    size = image_size

    dataset=ImageFolder(root=image_path,  image_size=image_size, mode=mode, is_distributed=is_distributed)

    collate_fn = collate_fn_launch

    data_loader=data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                sampler=None,
                                drop_last=True,
                                worker_init_fn=work_init_fn,
                                generator=generator,
                                collate_fn=collate_fn)
    return data_loader
