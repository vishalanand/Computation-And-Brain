import torch
import torchvision as tv
import os, sys, pickle
import numpy as np
from PIL import Image

def load_cfar10_batch_new(cifar10_dataset_folder_path, batch_name):
    #print(cifar10_dataset_folder_path, batch_name)

    with open(cifar10_dataset_folder_path + batch_name, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data']
    features = features.reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
        
    return features, labels


class ImagesDataset(torch.utils.data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True,
                 transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        #self.target_transform = target_transform
        self.train = train
        self.target_transform = None
        #return

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.cnt = 3000

        '''
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        '''

        self._load_meta()
        self.data_all = np.array([], dtype=np.float64).reshape(0, 32, 32, 3)
        self.labels_all = []

        for method in ['reverse']:
            list_iter = []
            if self.train:
                list_iter = ['data_batch_1']#, 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
                self.cnt = 3000
            else:
                list_iter = ['test_batch']
                self.cnt = 200

            
            for route in range(2, 3):
                for batch in list_iter:
                    arr, labels = load_cfar10_batch_new("./data_original_one/cifar-10-batches-py/", batch)
                    self.labels_all = self.labels_all + labels[:self.cnt]
                    for i in range(1, self.cnt+1):
                        #im = Image.open("./images/new/CIFAR_{}_{}_{}/{}.jpg".format(batch, method, route, i))
                        im = Image.open("./images/orig/CIFAR_{}/{}.jpg".format(batch, i))
                        im = (np.array(im))
                        #print(type(im), im.shape)
                        #self.data_all.append([im])
                        self.data_all = np.vstack((self.data_all, [im]))
                        '''
                        r = im[:,:,0].flatten()
                        g = im[:,:,1].flatten()
                        b = im[:,:,2].flatten()
                        abcd = np.array(list(r) + list(g) + list(b), dtype=np.float)
                        #print(abcd.shape)
                        self.data_all.append(abcd)
                        #self.labels_all.extend(labels[i - 1])
                        #print(arr.reshape(-1, 3, 32, 32))
                        #print(arr.shape)
                        #self.data_all = np.vstack((self.data_all, arr.reshape(-1, 3, 32, 32)))
                        '''
            
        #self.data_all = np.vstack(self.data_all).reshape(-1, 3, 32, 32)
        print(self.data_all.shape)
        #self.data_all = self.data_all.transpose((0, 2, 3, 1))
        #print(self.data_all.shape)
        #print(self.labels_all)

    def __getitem__(self, index):
        #return self.data_all[index], self.labels_all[index]

        # if self.train:
        #     return Image.open("./images/new/CIFAR_data_batch_{}_reverse_2/{}.jpg".format(index/10000 + 1, index%10000 + 1)), 
        # else:
        #     return Image.open("./images/new/CIFAR_test_batch_reverse_2/{}.jpg".format(index%10000 + 1))

        #img, target = self.data[index], self.targets[index]
        img, target = self.data_all[index], self.labels_all[index]
        #print(type(img))
        #img = Image.fromarray(img.astype(np.uint8))
        img = Image.fromarray(img.astype(np.uint8))
        #print(type(img))
        #return img, target

        if self.transform is not None:
            img = self.transform(img)
            #print(type(img), img.shape)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.cnt

        if(self.train):
            return 10000
        else:
            return 10000

        return len(self.data)
        print(self.labels_all.shape, "Shape")
        return len(self.labels_all.shape)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
