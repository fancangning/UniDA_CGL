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
    def __init__(self, image_list, transform=None, target_transform=None, return_paths=False, loader=default_loader, train=False, return_id=False, label_flag=None, return_label_flag=False):
        imgs, labels = make_dataset_nolist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_paths = return_paths
        self.return_id = return_id
        self.train = train
        self.return_label_flag = return_label_flag
        if self.return_label_flag:
            self.label_flag = label_flag
    
    def __getitem__(self, index):

        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)
        img = self.transform(img)
        if self.return_label_flag:
            flag = self.label_flag[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_paths:
            return img, target, path
        elif self.return_id:
            return img, target, index
        elif self.return_label_flag:
            return img, target, flag
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
        # self.target_ratio = 0 no mixup
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

        v = []
        num_class_index_target = int(self.target_ratio * (self.num_class - 1))

        if self.target_ratio > 0:
            available_index = [key for key in self.target_image_list.keys() if len(self.target_image_list[key]) > 0 
                               and key < self.num_class - 1]
            class_index_target = random.sample(available_index, min(num_class_index_target, len(available_index)))

        # class_index_source + class_index_target = list(set(range(self.num_class - 1)))
        class_index_source = list(set(range(self.num_class - 1)) - set(class_index_target))
        random.shuffle(class_index_source)

        for classes in class_index_source:
            # select support samples from source domain
            random_idx = random.randint(0, len(self.source_image[classes])-1)
            image = Image.open(self.source_image[classes][random_idx]).convert('RGB')

            if self.transformer is not None:
                image = self.transformer(image)
            image_data.append(image)
            label_data.append(classes)
            domain_label.append(1)
            ST_split.append(0)
            v.append(self.s_v_dict[classes][random_idx])
            # target_real_label.append(classes)
        for classes in class_index_target:
            # select support samples from target domain
            random_idx = random.randint(0, len(self.target_image_list[classes])-1)
            image = Image.open(self.target_image_list[classes][random_idx]).convert('RGB')

            if self.transformer is not None:
                image = self.transformer(image)
            image_data.append(image)
            label_data.append(classes)
            domain_label.append(0)
            ST_split.append(0)
            v.append(self.t_v_dict[classes][random_idx])
            # target_real_label.append(classes)
        
        # adding target samples
        for i in range(self.num_class - 1):

            if self.partition == 'train':
                if self.target_ratio > 0:
                    index = random.choice(list(range(len(self.label_flag))))
                else:
                    index = random.choice(list(range(len(self.target_image))))
                target_image = Image.open(self.target_image[index]).convert('RGB')
                if self.transformer is not None:
                    target_image = self.transformer(target_image)
                image_data.append(target_image)
                label_data.append(self.label_flag[index].item())
                target_real_label.append(self.target_label[index])
                domain_label.append(0)
                ST_split.append(1)
                v.append(self.t_v[index])
            elif self.partition == 'test':
                target_image = Image.open(self.target_image[item * (self.num_class - 1) + i]).convert('RGB')
                if self.transformer is not None:
                    target_image = self.transformer(target_image)
                image_data.append(target_image)
                label_data.append(self.num_class)
                target_real_label.append(self.target_label[item * (self.num_class - 1) + i])
                domain_label.append(0)
                ST_split.append(1)
                v.append(self.t_v[item * (self.num_class - 1) + i])
        image_data = torch.stack(image_data)
        label_data = torch.LongTensor(label_data)
        real_label_data = torch.tensor(target_real_label)
        domain_label = torch.tensor(domain_label)
        ST_split = torch.tensor(ST_split)
        v = torch.tensor(v)
        return image_data, label_data, real_label_data, domain_label, ST_split, v
    
    def load_dataset(self):
        if self.s_v is None:
            self.s_v = torch.zeros(len(open(self.source_path, 'r').readlines()))
        s_v_dict = {key: [] for key in range(self.num_class-1)}
        source_image_list = {key: [] for key in range(self.num_class-1)}
        target_image_list = []
        target_label_list = []
        with open(self.source_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = line.split(' ')
                label = label.strip()
                if label == str(self.num_class-1):
                    continue
                source_image_list[int(label)].append(image_dir)
                s_v_dict[int(label)].append(self.s_v[ind])
        
        with open(self.target_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = line.split(' ')
                label = label.strip()
                target_image_list.append(image_dir)
                target_label_list.append(int(label))
        
        return source_image_list, s_v_dict, target_image_list, target_label_list

class Office_Dataset(Base_Dataset):

    def __init__(self, root, partition, environment, s_v=None, t_v=None, label_flag=None, source='A', target='W', target_ratio=0.0, class_num=10):
        super(Office_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        src_name, tar_name = self.getFilePath(source, target, environment)
        self.source_path = os.path.join(root, src_name)
        self.target_path = os.path.join(root, tar_name)
        # self.class_name = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
        #                    'laptop_computer', 'monitor', 'mouse', 'mug', 'projector', 'unk']
        self.num_class = class_num + 1
        self.s_v = s_v
        # self.source_image is a dict, and self.target_image and self.target_label are lists.
        self.source_image, self.s_v_dict, self.target_image, self.target_label = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag
        self.t_v = t_v
        # create the unlabeled tag
        if self.label_flag is None:
            # self.label_flag initialization [11, 11, 11, ...]
            self.label_flag = torch.ones(len(self.target_image))*self.num_class
        else:
            # if pseudo labels come
            self.target_image_list = {key: [] for key in range(self.num_class+1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])
        # create the t_v_dict
        if self.t_v is None:
            # self.t_v initialization [0, 0, 0, ...]
            self.t_v = torch.zeros(len(self.target_image))
        else:
            # create the t_v_dict
            assert len(self.t_v) == len(self.label_flag)
            self.t_v_dict = {key: [] for key in range(self.num_class+1)}
            for i in range(len(self.t_v)):
                self.t_v_dict[self.label_flag[i].item()].append(self.t_v[i])
        
        if self.target_ratio > 0:
            self.alpha_value = [len(self.source_image[key])+len(self.target_image_list[key]) for key in self.source_image.keys()]
        else:
            self.alpha_value = self.alpha
        
        self.alpha_value = np.array(self.alpha_value)
        self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
        self.alpha_value = torch.tensor(self.alpha_value).float().cuda()
    
    def getFilePath(self, source, target, environment):
        if environment == 'oda':
            if source == 'A':
                src_name = 'source_amazon_oda.txt'
            elif source == 'W':
                src_name = 'source_webcam_oda.txt'
            elif source == 'D':
                src_name = 'source_dslr_oda.txt'
            else:
                print('Unknown Source Type, only supports A W D.')

            if target == 'A':
                tar_name = 'target_amazon_oda.txt'
            elif target == 'W':
                tar_name = 'target_webcam_oda.txt'
            elif target == 'D':
                tar_name = 'target_dslr_oda.txt'
            else:
                print('Unknown Target Type, only supports A W D.')
        elif environment == 'opda':
            if source == 'A':
                src_name = 'source_amazon_opda.txt'
            elif source == 'W':
                src_name = 'source_webcam_opda.txt'
            elif source == 'D':
                src_name = 'source_dslr_opda.txt'
            else:
                print('Unknown Source Type, only supports A W D.')

            if target == 'A':
                tar_name = 'target_amazon_opda.txt'
            elif target == 'W':
                tar_name = 'target_webcam_opda.txt'
            elif target == 'D':
                tar_name = 'target_dslr_opda.txt'
            else:
                print('Unknown Target Type, only supports A W D.')
        else:
            raise Exception('Unknown environment')
        
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
        target_folder = ImageFolder(self.target_path, transform=transformer, label_flag=self.label_flag, return_label_flag=True)
        source_loader = data.DataLoader(source_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
        target_loader = data.DataLoader(target_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

        s_score = list()
        t_score = list()
        pred_labels = list()

        model.eval()
        gnnModel.eval()
        discriminator_no_back.eval()
        for batch_idx, (img, label) in enumerate(source_loader):
            img, label = img.cuda(), label.cuda()
            img = img.unsqueeze(0)
            label = label.squeeze().unsqueeze(0)

            init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(label)

            with torch.no_grad():
                features = model(img)
                edge_logits, node_logits = gnnModel(init_node_feat=features, init_edge_feat=init_edge, target_mask=target_edge_mask)
                domain_pred = discriminator_no_back(features)

                norm_node_logits = F.softmax(node_logits[-1], dim=-1)
                score = torch.sum(-1*norm_node_logits*torch.log(norm_node_logits), -1)

                score = score / torch.log(torch.tensor([self.num_class - 1])).cuda() - domain_pred
                s_score.append(score[0].cpu().detach())
        
        for batch_idx, (img, label, flag) in enumerate(target_loader):
            img, label, flag = img.cuda(), label.cuda(), flag.cuda()
            img = img.unsqueeze(0)
            label = label.squeeze().unsqueeze(0)
            flag = flag.squeeze().unsqueeze(0)

            init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(flag)

            with torch.no_grad():
                features = model(img)
                edge_logits, node_logits = gnnModel(init_node_feat=features, init_edge_feat=init_edge, target_mask=target_edge_mask)
                domain_pred = discriminator_no_back(features)

                norm_node_logits = F.softmax(node_logits[-1], dim=-1)
                _, target_pred = norm_node_logits.max(-1)
                score = torch.sum(-1*norm_node_logits*torch.log(norm_node_logits), -1)

                score = domain_pred - score / torch.log(torch.tensor([self.num_class - 1])).cuda()
                t_score.append(score[0].cpu().detach())
                pred_labels.append(target_pred[0].cpu().detach())
        s_score = torch.cat(s_score)
        t_score = torch.cat(t_score)
        pred_labels = torch.cat(pred_labels)

        model.train()
        gnnModel.train()
        discriminator_no_back.train()
        
        print('s_score.size(): ', s_score.size())
        print('t_score.size(): ', t_score.size())
        print('max(s_score): ', max(s_score))
        print('min(s_score): ', min(s_score))
        print('max(t_score): ', max(t_score))
        print('min(t_score): ', min(t_score))
        print('pred_labels.size(): ', pred_labels.size())
        return s_score, t_score, pred_labels
