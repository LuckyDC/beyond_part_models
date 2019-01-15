import os

from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import ImageFolder
from data.dataset import ImageListFile
from transform.random_erase import RandomErase


def get_train_loader(root, batch_size, image_size, random_mirror=False, random_erase=False,
                     random_crop=False, num_workers=4):
    # data pre-processing
    aug_list = list()
    aug_list.append(transforms.Resize(image_size, interpolation=3))

    if random_mirror:
        aug_list.append(transforms.RandomHorizontalFlip())
    if random_crop:
        aug_list.append(transforms.RandomCrop(image_size))

    aug_list.append(transforms.ToTensor())
    aug_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if random_erase:
        aug_list.append(RandomErase(mean=[0.0, 0.0, 0.0]))

    transform = transforms.Compose(aug_list)

    # dataset
    train_dataset = ImageFolder(root, transform=transform, recursive=True, label_organize=True)

    # data loader
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True,
                              num_workers=num_workers)

    return train_loader


def get_test_loader(root, batch_size, image_size, num_workers=4):
    aug_list = list()
    aug_list.append(transforms.Resize(image_size, interpolation=3))
    aug_list.append(transforms.ToTensor())
    aug_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # transform
    transform = transforms.Compose(aug_list)

    # dataset
    if root.lower().find("msmt") != -1:
        prefix = os.path.join(os.path.dirname(root), "test")
        test_dataset = ImageListFile(root, prefix=prefix, transform=transform)
    else:
        test_dataset = ImageFolder(root, transform=transform)

    # dataloader
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False,
                             num_workers=num_workers)

    return test_loader
