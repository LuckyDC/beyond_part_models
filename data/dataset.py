import os
import torch
import numpy as np

from PIL import Image
from glob import glob
from torch.utils.data import Dataset

'''
    Specific dataset classes for person re-identification dataset. 
'''


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, ext="jpg", recursive=False, label_organize=False):
        if recursive:
            self.image_list = glob(os.path.join(root, "**", "*." + ext), recursive=recursive)
        else:
            self.image_list = glob(os.path.join(root, "*." + ext))
        self.image_list.sort()

        ids = []
        cam_ids = []
        for img_path in self.image_list:
            splits = os.path.basename(img_path).split("_")
            ids.append(int(splits[0]))

            if root.lower().find("msmt") != -1:
                cam_id = int(splits[2])
            else:
                cam_id = int(splits[1][1])

            cam_ids.append(cam_id)

        if label_organize:
            unique_ids = set(ids)
            label_map = dict(zip(unique_ids, range(len(unique_ids))))

            ids = map(lambda x: label_map[x], ids)
            ids = list(ids)

        self.cam_ids = cam_ids
        self.ids = ids

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_path = self.image_list[item]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(self.ids[item]), torch.tensor(self.cam_ids[item])


class ImageListFile(Dataset):
    def __init__(self, path, prefix=None, transform=None, label_organize=False):
        if os.path.isfile(path):
            raise ValueError("The file %s does not exist." % path)

        image_list = list(np.loadtxt(path, delimiter=" ", dtype=np.str)[:, 0])

        if prefix is not None:
            image_list = map(lambda x: os.path.join(prefix, x), image_list)
        self.image_list = list(image_list)
        self.image_list.sort()

        ids = []
        cam_ids = []
        for img_path in self.image_list:
            splits = os.path.basename(img_path).split("_")
            ids.append(int(splits[0]))

            if path.lower().find("msmt") != -1:
                cam_id = int(splits[2])
            else:
                cam_id = int(splits[1][1])

            cam_ids.append(cam_id)

        if label_organize:
            unique_ids = set(ids)
            label_map = dict(zip(unique_ids, range(len(unique_ids))))

            ids = map(lambda x: label_map[x], ids)
            ids = list(ids)

        self.cam_ids = cam_ids
        self.ids = ids

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_path = self.image_list[item]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(self.ids[item]), torch.tensor(self.cam_ids[item])
