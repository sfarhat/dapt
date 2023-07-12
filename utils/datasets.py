import json
import os
import sys
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import random_split, Dataset, Subset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision import datasets, transforms

# Credit to https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
class ImageNetKaggle(Dataset):

    def __init__(self, root, split, transform=None):

        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]

class CIFAR100Indices(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.cifar100 = datasets.CIFAR100(root, train, transform, target_transform, download)

    def __getitem__(self, i):
        data, target = self.cifar100[i]
        return data, target, i

    def __len__(self):
        return len(self.cifar100)

# Based off of https://github.com/HobbitLong/RepDistiller/blob/master/dataset/cifar100.py
class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = self.data.shape[0]
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):

                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner) 


            safe_extract(tar, path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            return img, target, idx

        return img, target

class CubInstanceSample(Cub2011):
    """
    CubInstance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 200
        # self.data is a pandas dataframe
        num_samples = self.data.shape[0]
        # labels all shifted up by 1
        label = self.data.target.values - 1

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        # self.cls_positive = np.asarray(self.cls_positive)
        # self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):

        sample = self.data.iloc[index]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if not self.is_sample:
            # directly return
            return img, target, index

        # sample contrastive examples
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)
        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return img, target, index, sample_idx

def find_classes(directory):
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class Scenes(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 train=True):
        self.train = train
        imagelist_file = 'Images.txt'
        if train:
            imagelist_file = 'Train'+imagelist_file
        else :
            imagelist_file = 'Test' + imagelist_file
        filesnames = open(os.path.join(root, imagelist_file)).read().splitlines()
        self.root = os.path.join(root, 'indoorCVPR_09/Images')
        classes, class_to_idx = find_classes(self.root)

        images = []

        for filename in list(set(filesnames)):
            target = filename.split('/')[0]
            path = os.path.join(root, 'indoorCVPR_09/Images/' + filename)
            item = (path, class_to_idx[target])
            images.append(item)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = images

        self.imgs = self.samples
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        if self.train:
            return sample, target, index
        else:
            return sample, target

class InstanceSampleScenes(Scenes):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 67
        self.data, self.targets = list(zip(*self.samples))
        num_samples = len(self.samples)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        # self.cls_positive = np.asarray(self.cls_positive)
        # self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        path, target = self.data[index], self.targets[index]

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx

class DTDIndices(Dataset):

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        self.dtd = datasets.DTD(root=root, split=split, download=download, transform=transform)

    def __getitem__(self, i):
        data, target = self.dtd[i]
        return data, target, i

    def __len__(self):
        return len(self.dtd)

class DTDInstanceSample(datasets.DTD):
    """
    DTDInstance+Sample Dataset
    """
    def __init__(self, root, split='train',
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):

        super().__init__(root=root, split=split, download=download, transform=transform)

        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 47
        num_samples = self.__len__()
        label = self._labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self._image_files[index], self._labels[index]
        img = Image.open(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index

        # sample contrastive examples
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)
        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return img, target, index, sample_idx

class Caltech101Dataset(Dataset):
    def __init__(self, subset, transform=None, train=True):
        self.subset = subset
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        x, y = self.subset[index]
        x = x.convert("RGB")
        if self.transform:
            x = self.transform(x)
        if self.train:
            return x, y, index
        else:
            return x, y

    def __len__(self):
        return len(self.subset)

class Caltech101InstanceSample(Caltech101Dataset):
    def __init__(self, subset, transform=None, 
                 k=4096, mode='exact', is_sample=True, percent=1.0):

        super().__init__(subset, transform=transform)

        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 101 
        num_samples = self.__len__()

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            _, y = self.subset[i]
            self.cls_positive[y].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.subset[index]
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index

        # sample contrastive examples
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)
        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return img, target, index, sample_idx

class CIFAR10Indices(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.cifar10 = datasets.CIFAR10(root, train, transform, target_transform, download)

    def __getitem__(self, i):
        data, target = self.cifar10[i]
        return data, target, i

    def __len__(self):
        return len(self.cifar10)

class CIFAR10InstanceSample(datasets.CIFAR10):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 10
        num_samples = self.data.shape[0]
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index

        # sample contrastive examples
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)
        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return img, target, index, sample_idx

class SyntheticDataset(Dataset):
    def __init__(self, syn_data_path, dataset, synset_size):

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.syn_train = ImageFolder(os.path.join(syn_data_path, f"{dataset} ({synset_size})"), transform=train_transform)

    def __getitem__(self, i):
        data, target = self.syn_train[i]
        return data, target, i

    def __len__(self):
        return len(self.syn_train)

def get_dataset(args):

    dataset = args.dataset
    data_path = args.data_path
    train_batch_size = args.train_bs

    if dataset == 'cifar100':
        # For imagenet backbone, it needs 224x224
        channel = 3
        num_classes = 100

        # ImageNet statistics
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # CIFAR-100 statistics
        # mean = [0.5071, 0.4866, 0.4409]
        # std = [0.2673, 0.2564, 0.2762]

        im_size = (224, 224)

        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomResizedCrop(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if 'distill' in vars(args) and args.distill == 'crd':
            train_set = CIFAR100InstanceSample(data_path, train=True, download=True, transform=train_transform, k=args.nce_k,
                                               mode=args.mode, is_sample=True)  # no augmentation

        else:
            # train_set = datasets.CIFAR100(data_path, train=True, download=True, transform=train_transform)  # no augmentation
            train_set = CIFAR100Indices(data_path, train=True, download=True, transform=train_transform)  # no augmentation

        test_set = datasets.CIFAR100(data_path, train=False, download=True, transform=test_transform)
        # class_names = train_set.classes

    elif dataset == 'imagenet':
        channel = 3
        im_size = (256, 256)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        num_classes = 1000
        transform = transforms.Compose(
                    [
                        transforms.Resize(im_size),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
        train_set = ImageNetKaggle(data_path, 'train', transform)
        test_set = ImageNetKaggle(data_path, 'val', transform)

    elif dataset == 'cub2011':
        channel = 3
        im_size = (224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        num_classes = 200
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if 'distill' in vars(args) and args.distill == 'crd':
            train_set = CubInstanceSample(data_path, train=True, download=True, transform=train_transform, k=args.nce_k,
                                               mode=args.mode, is_sample=True)  # no augmentation

        else:
            train_set = Cub2011(data_path, train=True, transform=train_transform, download=True)

        test_set = Cub2011(data_path, train=False, transform=test_transform, download=True)

    elif dataset == 'mit_indoor':
        channel = 3
        im_size = (224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        num_classes = 67 
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if 'distill' in vars(args) and args.distill == 'crd':
            train_set = InstanceSampleScenes(os.path.join(data_path, 'mit'), train=True, transform=train_transform, k=args.nce_k,
                                               mode=args.mode, is_sample=True) 
        else:
            train_set = Scenes(os.path.join(data_path, 'mit'), transform=train_transform, train=True) 

        test_set = Scenes(os.path.join(data_path, 'mit'), transform=test_transform, train=False)

    elif dataset == 'dtd':
        channel = 3
        im_size = (224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        num_classes = 47 
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if 'distill' in vars(args) and args.distill == 'crd':
            train_set = DTDInstanceSample(data_path, split='train', transform=train_transform, download=True, 
                                          k=args.nce_k, mode=args.mode, is_sample=True)
        else:
            # train_set = torchvision.datasets.DTD(os.path.join(data_path, 'dtd'), split='train', transform=train_transform, download=True)
            train_set = DTDIndices(os.path.join(data_path, 'dtd'), split='train', transform=train_transform, download=True)

        test_set = torchvision.datasets.DTD(os.path.join(data_path, 'dtd'), split='test', transform=test_transform, download=True)

    elif dataset == 'caltech101':
        channel = 3
        im_size = (224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        num_classes = 101 
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        ds = torchvision.datasets.Caltech101(os.path.join(data_path, 'caltech101'), download=True)
        class_start_idx = [0] + [i for i in range(1, len(ds)) if ds.y[i] == ds.y[i-1] + 1]

        # Since the classes are imbalanced, we grab 80% of each class' images from the train indices
        class_len = [class_start_idx[i+1] - class_start_idx[i] for i in range(len(class_start_idx) - 1)]
        # Edge case
        class_len.append(len(ds) - class_start_idx[-1])

        train_indices = sum([list(range(class_start_idx[i], class_start_idx[i] + int(.8 * class_len[i]))) for i in range(len(class_start_idx))], [])
        test_indices = list((set(range(1, len(ds))) - set(train_indices) ))

        train_subset = Subset(ds, train_indices)
        test_subset = Subset(ds, test_indices)

        if 'distill' in vars(args) and args.distill == 'crd':
            train_set = Caltech101InstanceSample(train_subset, transform=train_transform, k=args.nce_k, mode=args.mode, is_sample=True)
        else:
            train_set = Caltech101Dataset(train_subset, train_transform, train=True)

        test_set = Caltech101Dataset(test_subset, transform=test_transform, train=False)

    elif dataset == 'cifar10':
        # For imagenet backbone, it needs 224x224
        channel = 3
        #im_size = (32, 32)
        im_size = (224, 224)
        num_classes = 10

        # ImageNet statistics
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # CIFAR-10 statistics
        # mean = [0.4914, 0.4822, 0.4465]
        # std = [0.2023, 0.1994, 0.2010]

        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomResizedCrop(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if 'distill' in vars(args) and args.distill == 'crd':
            train_set = CIFAR10InstanceSample(data_path, train=True, download=True, transform=train_transform, k=args.nce_k,
                                               mode=args.mode, is_sample=True)

        else:
            # train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transform)  
            train_set = CIFAR10Indices(data_path, train=True, download=True, transform=train_transform)  

        test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transform)

    else:
        sys.exit(f'unknown dataset: {dataset}')

    # Creating validation sets (20% of training set == 10% of total set)
    # Fwiw, this doens't class balance
    # TODO: this is broken if we're using the CRD loader, so we'll get rid of it for now
    split = 1
    train_set, val_set = random_split(train_set, [int(split * len(train_set)), len(train_set) - int(split * len(train_set))])

    test_batch_size = args.test_bs

    trainloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    valloader = DataLoader(val_set, batch_size=test_batch_size, shuffle=False)
    testloader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

    return train_set, trainloader, val_set, valloader, test_set, testloader, channel, num_classes, im_size
