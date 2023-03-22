from __future__ import print_function
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import random
import os
import os.path
import numpy as np
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset_nolist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list

class ImageFolder(data.Dataset):
    def __init__(self, image_list, transform=None, target_transform=None, return_paths=False, loader=default_loader, train=False, return_id=False):
        imgs, labels = make_dataset_nolist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_paths = return_paths
        self.return_id = return_id
        self.train = train
    
    def __getitem__(self, index):

        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)
        img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_paths:
            return img, target, path
        elif self.return_id:
            return img, target, index
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs)

class Base_Dataset(data.Dataset):
    def __init__(self, root, partition, target_ratio=0.0):
        super(Base_Dataset, self).__init__()
        # set dataset info
        self.root = root
        self.partition = partition
        self.target_ratio = target_ratio
        # self.target_ratio=0 no mixup
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.Resize(256),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomCrop(224),
                                                   transforms.ToTensor(),
                                                   normalize])
        else:
            self.transformer = transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   normalize])

    def __len__(self):

        if self.partition == 'train':
            return int(min(sum(self.alpha), len(self.target_image)) / (self.num_class - 1))
        elif self.partition == 'test':
            return int(len(self.target_image) / (self.num_class - 1))

    def __getitem__(self, item):

        image_data = []
        label_data = []

        target_real_label = []
        class_index_target = []

        domain_label = []
        ST_split = [] # Mask of targets to be evaluated
        # select index for support class
        num_class_index_target = int(self.target_ratio * (self.num_class - 1))

        if self.target_ratio > 0:
            available_index = [key for key in self.target_image_list.keys() if len(self.target_image_list[key]) > 0
                               and key < self.num_class - 1]
            class_index_target = random.sample(available_index, min(num_class_index_target, len(available_index)))

        # class_index_source + class_index_target = list(set(range(self.num_class - 1)))
        class_index_source = list(set(range(self.num_class - 1)) - set(class_index_target))
        random.shuffle(class_index_source)

        for classes in class_index_source:
            # select support samples from source domain or target domain
            # print('classes:', classes)
            # print('type(self.source_image[classes]):', type(self.source_image[classes]))
            # print('len(self.source_image[classes]):', (self.source_image[classes]))
            image = Image.open(random.choice(self.source_image[classes])).convert('RGB')

            if self.transformer is not None:
                image = self.transformer(image)
            image_data.append(image)
            label_data.append(classes)
            domain_label.append(1)
            ST_split.append(0)
            # target_real_label.append(classes)
        for classes in class_index_target:
            # select support samples from source domain or target domain
            image = Image.open(random.choice(self.target_image_list[classes])).convert('RGB')

            if self.transformer is not None:
                image = self.transformer(image)
            image_data.append(image)
            label_data.append(classes)
            domain_label.append(0)
            ST_split.append(0)
            # target_real_label.append(classes)

        # adding target samples
        for i in range(self.num_class - 1):

            if self.partition == 'train':
                if self.target_ratio > 0:
                    index = random.choice(list(range(len(self.label_flag))))
                else:
                    index = random.choice(list(range(len(self.target_image))))
                # index = random.choice(list(range(len(self.label_flag))))
                target_image = Image.open(self.target_image[index]).convert('RGB')
                if self.transformer is not None:
                    target_image = self.transformer(target_image)
                image_data.append(target_image)
                # print('self.label_flag[index]:', self.label_flag[index])
                label_data.append(self.label_flag[index].item())
                target_real_label.append(self.target_label[index])
                domain_label.append(0)
                ST_split.append(1)
            elif self.partition == 'test':
                # For last batch
                # if item * (self.num_class - 1) + i >= len(self.target_image):
                #     break
                target_image = Image.open(self.target_image[item * (self.num_class - 1) + i]).convert('RGB')
                if self.transformer is not None:
                    target_image = self.transformer(target_image)
                image_data.append(target_image)
                label_data.append(self.num_class)
                target_real_label.append(self.target_label[item * (self.num_class - 1) + i])
                domain_label.append(0)
                ST_split.append(1)
        image_data = torch.stack(image_data)
        # print('label_data:', label_data)
        label_data = torch.LongTensor(label_data)
        real_label_data = torch.tensor(target_real_label)
        domain_label = torch.tensor(domain_label)
        ST_split = torch.tensor(ST_split)
        return image_data, label_data, real_label_data, domain_label, ST_split

    def load_dataset(self):
        source_image_list = {key: [] for key in range(self.num_class - 1)}
        target_image_list = []
        target_label_list = []
        with open(self.source_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = line.split(' ')
                label = label.strip()
                if label == str(self.num_class-1):
                    continue
                source_image_list[int(label)].append(image_dir)
                # source_image_list.append(image_dir)

        with open(self.target_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = line.split(' ')
                label = label.strip()
                # target_image_list[int(label)].append(image_dir)
                target_image_list.append(image_dir)
                target_label_list.append(int(label))

        return source_image_list, target_image_list, target_label_list


class Office_Dataset(Base_Dataset):

    def __init__(self, root, partition, label_flag=None, source='A', target='W', target_ratio=0.0):
        super(Office_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join(root, src_name)
        self.target_path = os.path.join(root, tar_name)
        self.class_name = ["back_pack", "bike", "calculator", "headphones", "keyboard", 
                            "laptop_computer", "monitor", "mouse", "mug", "projector", "unk"]
        self.num_class = len(self.class_name)
        # self.source_image is a dict, and self.target_image and self.target_label are lists.
        self.source_image, self.target_image, self.target_label = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag
        
        # print('label_flag: ', label_flag)
        # create the unlabeled tag
        if self.label_flag is None:
            # self.label_flag initialization [11, 11, 11, ...]
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])

        if self.target_ratio > 0:
            self.alpha_value = [len(self.source_image[key]) + len(self.target_image_list[key]) for key in self.source_image.keys()]
        else:
            self.alpha_value = self.alpha

        self.alpha_value = np.array(self.alpha_value)
        self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
        self.alpha_value = torch.tensor(self.alpha_value).float().cuda()

    def getFilePath(self, source, target):

        if source == 'A':
            src_name = 'source_amazon_oda.txt'
        elif source == 'W':
            src_name = 'source_webcam_oda.txt'
        elif source == 'D':
            src_name = 'source_dslr_oda.txt'
        else:
            print("Unknown Source Type, only supports A W D.")

        if target == 'A':
            tar_name = 'target_amazon_oda.txt'
        elif target == 'W':
            tar_name = 'target_webcam_oda.txt'
        elif target == 'D':
            tar_name = 'target_dslr_oda.txt'
        else:
            print("Unknown Target Type, only supports A W D.")

        return src_name, tar_name
    
    def label2edge(self, targets):
        
        # targets.size(): [1, 80]
        batch_size, num_sample = targets.size()
        # print('targets: ', targets)
        target_node_mask = torch.eq(targets, self.num_class).type(torch.bool).cuda()
        source_node_mask = ~target_node_mask & ~torch.eq(targets, self.num_class - 1).type(torch.bool)
        
        # label_i.size(): [1, 80, 80]
        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        # print('label_i.size(): ', label_i.size())
        # print('label_i: ', label_i)
        label_j = label_i.transpose(1, 2)
        # print('label_j: ', label_j)

        edge = torch.eq(label_i, label_j).float().cuda()
        target_edge_mask = (torch.eq(label_i, self.num_class) + torch.eq(label_j, self.num_class)).type(torch.bool).cuda()
        # print(torch.eq(label_i, self.num_class))
        # print(torch.eq(label_j, self.num_class))
        # print(torch.eq(label_i, self.num_class) + torch.eq(label_j, self.num_class))
        source_edge_mask = ~target_edge_mask
        init_edge = edge*source_edge_mask.float()

        return init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask

    def transferability_score(self, model, gnnModel, discriminator_no_back):
        
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        transformer = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=mean_pix, std=std_pix)])
        source_folder = ImageFolder(self.source_path, transform=transformer)
        target_folder = ImageFolder(self.target_path, transform=transformer)
        source_loader = data.DataLoader(source_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
        target_loader = data.DataLoader(target_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

        s_score = list()
        t_score = list()

        model.eval()
        gnnModel.eval()
        discriminator_no_back.eval()
        for batch_idx, (img, label) in enumerate(source_loader):
            img, label = img.cuda(), label.cuda()
            label = label.squeeze().unsqueeze(0)

            init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(label)

            with torch.no_grad():
                features = model(img)
                features = features.unsqueeze(0)
                edge_logits, node_logits = gnnModel(init_node_feat=features, init_edge_feat=init_edge, target_mask=target_edge_mask)
                domain_pred = discriminator_no_back(features)

                norm_node_logits = F.softmax(node_logits[-1], dim=-1)
                score = torch.sum(-1*norm_node_logits+torch.log(norm_node_logits), -1)

                score = score / (self.num_class - 1) - domain_pred
                s_score.append(score.cpu().detach())
        
        for batch_idx, (img, label) in enumerate(target_loader):
            img, label = img.cuda(), label.cuda()
            label = label.squeeze().unsqueeze(0)

            init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(label)

            with torch.no_grad():
                features = model(img)
                features = features.unsqueeze(0)
                edge_logits, node_logits = gnnModel(init_node_feat=features, init_edge_feat=init_edge, target_mask=target_edge_mask)
                domain_pred = discriminator_no_back(features)

                norm_node_logits = F.softmax(node_logits[-1], dim=-1)
                score = torch.sum(-1*norm_node_logits+torch.log(norm_node_logits), -1)

                score = domain_pred - score / (self.num_class - 1)
                t_score.append(score.cpu().detach())
        
        s_score = torch.cat(s_score)
        t_score = torch.cat(t_score)

        model.train()
        gnnModel.train()
        discriminator_no_back.train()

        return s_score, t_score



class Home_Dataset(Base_Dataset):
    def __init__(self, root, partition, label_flag=None, source='A', target='R', target_ratio=0.0):
        super(Home_Dataset, self).__init__(root, partition, target_ratio)
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join(root, src_name)
        self.target_path = os.path.join(root, tar_name)
        self.class_name = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                           'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                           'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
                           'Fork', 'unk']
        self.num_class = len(self.class_name)

        self.source_image, self.target_image, self.target_label = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag
        if self.label_flag is None:
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])

        # if self.target_ratio > 0:
        #     self.alpha_value = [len(self.source_image[key]) + len(self.target_image_list[key]) for key in
        #                         self.source_image.keys()]
        # else:
        #     self.alpha_value = self.alpha
        #
        # self.alpha_value = np.array(self.alpha_value)
        # self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
        # self.alpha_value = torch.tensor(self.alpha_value).float().cuda()

    def getFilePath(self, source, target):

        if source == 'A':
            src_name = 'art_source.txt'
        elif source == 'C':
            src_name = 'clip_source.txt'
        elif source == 'P':
            src_name = 'product_source.txt'
        elif source == 'R':
            src_name = 'real_source.txt'
        else:
            print("Unknown Source Type, only supports A C P R.")

        if target == 'A':
            tar_name = 'art_tar.txt'
        elif target == 'C':
            tar_name = 'clip_tar.txt'
        elif target == 'P':
            tar_name = 'product_tar.txt'
        elif target == 'R':
            tar_name = 'real_tar.txt'
        else:
            print("Unknown Target Type, only supports A C P R.")

        return src_name, tar_name


class Visda_Dataset(Base_Dataset):
    def __init__(self, root, partition, label_flag=None, target_ratio=0.0):
        super(Visda_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        self.source_path = os.path.join(root, 'source_list.txt')
        self.target_path = os.path.join(root, 'target_list.txt')
        self.class_name = ["bicycle", "bus", "car", "motorcycle", "train", "truck", 'unk']
        self.num_class = len(self.class_name)
        self.source_image, self.target_image, self.target_label = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag
        if self.label_flag is None:
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])

class Visda18_Dataset(Base_Dataset):
    def __init__(self, root, partition, label_flag=None, target_ratio=0.0):
        super(Visda18_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        self.source_path = os.path.join(root, 'source_list_k.txt')
        self.target_path = os.path.join(root, 'target_list.txt')
        self.class_name = ["areoplane","bicycle", "bus", "car", "horse", "knife", "motorcycle", "person", "plant",
                           "skateboard", "train", "truck", 'unk']
        self.num_class = len(self.class_name)
        self.source_image, self.target_image, self.target_label = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag
        if self.label_flag is None:
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])
