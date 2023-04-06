#********************************************#
# ScriptName: TransferabilityScoreAnalysis.py
# Author: fancangning@gmail.com
# Create Date: 2023-03-31 14:35
# Modify Author: fancangning@gmail.com
# Modify Date: 2023-03-31 14:35
# Function: Analysis the Transferability score to find the best way to discriminate target private classes and source private classes
#********************************************#

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import models
import data_loader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from sklearn import metrics
import argparse

from data_loader import ImageFolder
from models.component import Discriminator

parser = argparse.ArgumentParser(description='Analysis the Transferability score to find the best way to discriminate target private classes')
parser.add_argument('--source_data', type=str, default='./txt/office31/source_amazon_oda.txt', help='path to source list')
parser.add_argument('--target_data', type=str, default='./txt/office31/target_webcam_oda.txt', help='path to target list')
parser.add_argument('--arch', type=str, default='res')
parser.add_argument('--in_features', type=int, default=2048)
parser.add_argument('--gpu_devices', type=int, nargs='+', default=0, help='which gpu to use')
parser.add_argument('--output_path', type=str, default='snapshot', help='output path')
parser.add_argument('--model_path', type=str, default='./checkpoints000/D-office_src-A_tar-W_A-res_L-1_E-5_B-4_step_19.pth.tar')
parser.add_argument('--class_num', type=int, default=10, help='the number of source classes')
# node_features edge_features num_layers dropout
parser.add_argument('--node_features', type=int, default=1024)
parser.add_argument('--edge_features', type=int, default=1024)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.2)
args = parser.parse_args()

gpu_devices = str(args.gpu_devices)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices

def draw_distribution(source_statistics, normal_statistics, abnormal_statistics, distribution_type):
    max_length = max(len(source_statistics), len(normal_statistics), len(abnormal_statistics))

    if len(normal_statistics) == max_length:
        abnormal_statistics = torch.cat([abnormal_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(abnormal_statistics)))])
        source_statistics = torch.cat([source_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(source_statistics)))])
    elif len(abnormal_statistics) == max_length:
        normal_statistics = torch.cat([normal_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(normal_statistics)))])
        source_statistics = torch.cat([source_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(source_statistics)))])
    elif len(source_statistics) == max_length:
        normal_statistics = torch.cat([normal_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(normal_statistics)))])
        abnormal_statistics = torch.cat([abnormal_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(abnormal_statistics)))])
    else:
        raise Exception('wrong length')

    df = pd.DataFrame(
        {
            'target_common': normal_statistics,
            'target_private': abnormal_statistics,
            'source': source_statistics,
        }
    )
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
    fig.savefig(os.path.join(distribution_type+'.png'))
    return


def label2edge(targets, class_num):
        
    # targets.size(): [1, 80]
    batch_size, num_sample = targets.size()
    # print('targets: ', targets)
    target_node_mask = torch.eq(targets, class_num+1).type(torch.bool).cuda()
    source_node_mask = ~target_node_mask & ~torch.eq(targets, class_num).type(torch.bool)
    
    # label_i.size(): [1, 80, 80]
    label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
    # print('label_i.size(): ', label_i.size())
    # print('label_i: ', label_i)
    label_j = label_i.transpose(1, 2)
    # print('label_j: ', label_j)

    edge = torch.eq(label_i, label_j).float().cuda()
    target_edge_mask = (torch.eq(label_i, class_num+1) + torch.eq(label_j, class_num+1)).type(torch.bool).cuda()
    # print(torch.eq(label_i, self.num_class))
    # print(torch.eq(label_j, self.num_class))
    # print(torch.eq(label_i, self.num_class) + torch.eq(label_j, self.num_class))
    source_edge_mask = ~target_edge_mask
    init_edge = edge*source_edge_mask.float()

    return init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask


def main():
    args.num_class = args.class_num + 1
    model_path = args.model_path
    output_path = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)   
    log_file = open(os.path.join(output_path, model_name.split('.')[0]+'_'+args.source_data.split('_')[1]+'_'+args.target_data.split('_')[1]+'_Transferability_score.txt'), 'w')
    log_file.write(str(vars(args))+'\n')
    log_file.flush()

    # set the datasets
    mean_pix = [0.485, 0.456, 0.406]
    std_pix = [0.229, 0.224, 0.225]
    transformer = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean_pix, std=std_pix)])
    
    source_estimator_folder = ImageFolder(args.source_data, transform=transformer, return_paths=False)
    target_estimator_folder = ImageFolder(args.target_data, transform=transformer, return_paths=False)
    source_folder = ImageFolder(args.source_data, transform=transformer, return_paths=True)
    target_folder = ImageFolder(args.target_data, transform=transformer, return_paths=True)

    source_estimator_loader = torch.utils.data.DataLoader(source_estimator_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    target_estimator_loader = torch.utils.data.DataLoader(target_estimator_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    source_loader = torch.utils.data.DataLoader(source_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    target_loader = torch.utils.data.DataLoader(target_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

    # states = {'model': self.model.state_dict(),
    #               'graph': self.gnnModel.state_dict(),
    #               'discriminator': self.discriminator.state_dict(),
    #               'discriminator_no_back': self.discriminator_no_back.state_dict()}
    # torch.save(states, osp.join(args.checkpoints_dir, '{}_step_{}.pth.tar'.format(args.experiment, step)))
    # set the models
    model = models.create(args.arch, args)
    model = nn.DataParallel(model).cuda()
    gnnModel = models.create('gnn', args)
    gnnModel = nn.DataParallel(gnnModel).cuda()
    discriminator = Discriminator(args.in_features)
    discriminator = nn.DataParallel(discriminator).cuda()
    discriminator_no_back = Discriminator(args.in_features)
    discriminator_no_back = nn.DataParallel(discriminator_no_back).cuda()

    print('loading weight')
    state = torch.load(args.model_path)
    # print(model)
    # for key in state['model'].keys():
    #     print(key)
    # input()
    model.load_state_dict(state['model'])
    gnnModel.load_state_dict(state['graph'])
    discriminator.load_state_dict(state['discriminator'])
    discriminator_no_back.load_state_dict(state['discriminator_no_back'])

    s_score = list()
    t_score = list()
    labels = list()

    model.eval()
    gnnModel.eval()
    discriminator_no_back.eval()
    for batch_idx, (img, label) in enumerate(source_estimator_loader):
        img, label = img.cuda(), label.cuda()
        img = img.unsqueeze(0)
        label = label.squeeze().unsqueeze(0)

        init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = label2edge(label, args.class_num)
        
        with torch.no_grad():
            features = model(img)
            edge_logits, node_logits = gnnModel(init_node_feat=features, init_edge_feat=init_edge, target_mask=target_edge_mask)
            domain_pred = discriminator_no_back(features)

            norm_node_logits = F.softmax(node_logits[-1], dim=-1)
            score = torch.sum(-1*norm_node_logits*torch.log(norm_node_logits), -1)

            score = score / torch.log(torch.tensor([args.class_num])).cuda() - domain_pred
            s_score.append(score[0].cpu().detach())
    

    for batch_idx, (img, label) in enumerate(target_estimator_loader):
        img, label = img.cuda(), label.cuda()
        img = img.unsqueeze(0)
        label = label.squeeze().unsqueeze(0)

        init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = label2edge(label, args.class_num)

        with torch.no_grad():
            features = model(img)
            edge_logits, node_logits = gnnModel(init_node_feat=features, init_edge_feat=init_edge, target_mask=target_edge_mask)
            domain_pred = discriminator_no_back(features)

            norm_node_logits = F.softmax(node_logits[-1], dim=-1)
            _, target_pred = norm_node_logits.max(-1)
            score = torch.sum(-1*norm_node_logits*torch.log(norm_node_logits), -1)

            score = domain_pred - score / torch.log(torch.tensor([args.class_num])).cuda()
            t_score.append(score[0].cpu().detach())
            labels.append(label.cpu().detach().squeeze())
    s_score = torch.cat(s_score)
    t_score = torch.cat(t_score)
    labels = torch.cat(labels)

    draw_distribution(s_score, t_score[labels<10], t_score[labels==10], model_name.split('.')[0]+'_transferability_score')


if __name__ == '__main__':
    main()