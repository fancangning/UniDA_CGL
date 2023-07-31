import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import models
from torch.utils.data import DataLoader

import os
import os.path as osp
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from utils.logger import AverageMeter as meter
from data_loader import Office_Dataset, ImageFolder
from utils.loss import FocalLoss

from models.component import Discriminator

class ModelTrainer():
    def __init__(self, args, data, step=0, label_flag=None, s_v=None, t_v=None, logger=None):
        self.args = args
        self.batch_size = args.batch_size
        self.data_workers = 6

        self.step = step
        self.data = data
        self.label_flag = label_flag

        self.num_class = data.num_class
        self.num_task = args.batch_size
        self.s_num_to_select = 0
        self.t_num_to_select = 0

        self.model = models.create(args.arch, args)
        self.model = nn.DataParallel(self.model).cuda()

        # GNN
        self.gnnModel = models.create('gnn', args)
        self.gnnModel = nn.DataParallel(self.gnnModel).cuda()

        self.meter = meter(args.num_class)
        self.s_v = s_v
        self.t_v = t_v

        # CE for node classification
        if args.loss == 'focal':
            self.criterionCE = FocalLoss().cuda()
        elif args.loss == 'nll':
            self.criterionCE = nn.NLLLoss(reduction='mean').cuda()
        
        # BCE for edge
        self.criterion = nn.BCELoss(reduction='mean').cuda()
        self.global_step = 0
        self.val_acc = 0
        self.s_threshold = args.s_threshold
        self.t_threshold = args.t_threshold

        # Discriminator
        if self.args.discriminator:
            self.discriminator = Discriminator(self.args.in_features)
            self.discriminator = nn.DataParallel(self.discriminator).cuda()
            self.discriminator_no_back = Discriminator(self.args.in_features)
            self.discriminator_no_back = nn.DataParallel(self.discriminator_no_back).cuda()

    def get_dataloader(self, dataset, training=False):

        if self.args.visualization:
            data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_workers,
                                     shuffle=training, pin_memory=True, drop_last=True)
            return data_loader

        data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_workers,
                                 shuffle=training, pin_memory=True, drop_last=training)
        return data_loader
    
    def adjust_lr(self, epoch, step_size):
        lr = self.args.lr / (2 **(epoch // step_size))
        for g in self.optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        
        if epoch % step_size == 0:
            print('Epoch {}, current lr {}'.format(epoch, lr))
    
    def label2edge(self, targets):

        batch_size, num_sample = targets.size()
        target_node_mask = torch.eq(targets, self.num_class).type(torch.bool).cuda()
        source_node_mask = ~target_node_mask & ~torch.eq(targets, self.num_class-1).type(torch.bool)

        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)

        edge = torch.eq(label_i, label_j).float().cuda()
        target_edge_mask = (torch.eq(label_i, self.num_class) + torch.eq(label_j, self.num_class)).type(torch.bool).cuda()
        source_edge_mask = ~target_edge_mask
        init_edge = edge*source_edge_mask.float()

        return init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask

    def transform_shape(self, tensor):

        batch_size, num_class, other_dim = tensor.shape
        tensor = tensor.view(1, batch_size*num_class, other_dim)
        return tensor
    
    def train(self, step, epochs=70, step_size=55):
        args = self.args

        train_loader = self.get_dataloader(self.data, training=True)

        # initialize model

        # change the learning rate
        if args.arch == 'res':
            if args.dataset == 'office':
                param_groups = [
                    {'params': self.model.module.CNN.parameters(), 'lr_mult': 0.01},
                    {'params': self.gnnModel.parameters(), 'lr_mult': 0.1},
                ]
                if self.args.discriminator:
                    param_groups.append({'params': self.discriminator.parameters(), 'lr_mult': 0.1})
                    param_groups.append({'params': self.discriminator_no_back.parameters(), 'lr_mult': 0.1})
            else:
                raise Exception('Wrong architecture!')
            
            args.in_features = 2048
        else:
            raise Exception('Wrong architecture!')

        self.optimizer = torch.optim.Adam(params=param_groups, lr=args.lr, weight_decay=args.weight_decay)
        self.model.train()
        self.gnnModel.train()
        self.discriminator.train()
        self.discriminator_no_back.train()
        self.meter.reset()

        for epoch in range(epochs):
            self.adjust_lr(epoch, step_size)

            with tqdm(total=len(train_loader)) as pbar:
                for i, inputs in enumerate(train_loader):

                    # images.size(): [4, 20, 3, 224, 224] targets.size(): [4, 20]
                    images = Variable(inputs[0], requires_grad=False).cuda()
                    targets = Variable(inputs[1]).cuda()

                    # targets_DT.size(): [40]
                    targets_DT = targets[:, args.num_class-1:].reshape(-1)

                    if self.args.discriminator:
                        # domain_label.size(): [4, 20]
                        domain_label = Variable(inputs[3].float()).cuda()
                    
                    # selected_idx.size(): [4, 20]
                    selected_idx = inputs[5]

                    # targets.size() [4, 20]-> [4, 20, 1] -> [1, 80, 1] -> [1, 80]
                    targets = self.transform_shape(targets.unsqueeze(-1)).squeeze(-1)

                    # init_edge.size(): [1, 80, 80] target_edge_mask.size(): [1, 80, 80] source_edge_mask.size(): [1, 80, 80] 
                    # target_node_mask.size(): [1, 80] source_node_mask.size(): [1, 80]
                    init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(targets)

                    # features.size(): [4, 20, 2048]
                    features = self.model(images)
                    # features.size(): [4, 20, 2048] -> [1, 80, 2048]
                    features = self.transform_shape(features)

                    # edge_logits[0].size(): [1, 80, 80] node_logits[0].size(): [1, 80, 10]
                    edge_logits, node_logits = self.gnnModel(init_node_feat=features, init_edge_feat=init_edge, target_mask=target_edge_mask)

                    full_edge_loss = [self.criterion(edge_logit.masked_select(source_edge_mask), init_edge.masked_select(source_edge_mask)) for edge_logit in edge_logits]
                    # norm_node_logits.size(): [1, 80, 10]
                    norm_node_logits = F.softmax(node_logits[-1], dim=-1)

                    if args.loss == 'nll':
                        source_node_loss = self.criterionCE(torch.log(norm_node_logits[source_node_mask, :]+1e-5),
                                                            targets.masked_select(source_node_mask))
                    elif args.loss == 'focal':
                        source_node_loss = self.criterionCE(norm_node_logits[source_node_mask, :],
                                                            targets.masked_select(source_node_mask))
                    edge_loss = 0
                    for l in range(args.num_layers - 1):
                        edge_loss += full_edge_loss[l] * 0.5
                    edge_loss += full_edge_loss[-1] * 1

                    loss = 1 * edge_loss + args.node_loss * source_node_loss

                    if self.args.discriminator:
                        unk_label_mask = torch.eq(targets, args.num_class-1).squeeze()
                        domain_pred = self.discriminator(features)
                        domain_pred_no_back = self.discriminator_no_back(x=features, eta=0.0)
                        temp = torch.squeeze(domain_pred)[~unk_label_mask]
                        # temp.size(): [80] domain_label.view(-1)[~unk_label_mask].size(): 
                        w = selected_idx.view(-1)*9+1
                        domain_loss_function = nn.BCELoss(weight=w).cuda()
                        domain_loss = domain_loss_function(temp, domain_label.view(-1)[~unk_label_mask])
                        domain_loss_no_back = self.criterion(torch.squeeze(domain_pred_no_back), domain_label.view(-1))
                        loss = loss + args.adv_coeff * (domain_loss + domain_loss_no_back)
                    

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    pbar.update()
                    if i > 150:
                        break
        
        # save model
        states = {'model': self.model.state_dict(),
                  'graph': self.gnnModel.state_dict(),
                  'discriminator': self.discriminator.state_dict(),
                  'discriminator_no_back': self.discriminator_no_back.state_dict()}
        torch.save(states, osp.join(args.checkpoints_dir, '{}_step_{}.pth.tar'.format(args.experiment, step)))
        self.meter.reset()
    
    def select_top_data(self, s_score, t_score, step):
        # select a set of transferable source and target samples to adapt
        self.s_num_to_select = int(s_score.size()[0] / (100//self.args.EF))
        self.t_num_to_select = int(t_score.size()[0] / (100//self.args.EF))

        if self.s_v is None:
            self.s_v = np.zeros(s_score.size()[0])
        if self.t_v is None:
            self.t_v = np.zeros(t_score.size()[0])
        
        s_unselected_idx = np.where(self.s_v==0)[0]
        t_unselected_idx = np.where(self.t_v==0)[0]

        if max(s_score[s_unselected_idx]) > self.args.s_threshold[step]:
            if len(s_unselected_idx) < self.s_num_to_select:
                self.s_num_to_select = len(s_unselected_idx)
            index = np.argsort(-s_score[s_unselected_idx])
            index_orig = s_unselected_idx[index]
            num_pos = int(self.s_num_to_select)
            for i in range(num_pos):
                self.s_v[index_orig[i]] = 1
            # print('s_score[index_orig]: ', s_score[index_orig])
        
        if max(t_score[t_unselected_idx]) > self.args.t_threshold[step]:
            if len(t_unselected_idx) < self.t_num_to_select:
                self.t_num_to_select = len(t_unselected_idx)
            index = np.argsort(-t_score[t_unselected_idx])
            index_orig = t_unselected_idx[index]
            num_pos = int(self.t_num_to_select)
            for i in range(num_pos):
                self.t_v[index_orig[i]] = 1
            # print('t_score[index_orig]: ', t_score[index_orig])
        # print('s_score[self.s_v==1].size(): ', s_score[self.s_v==1].size())
        # print('s_score[self.s_v==1]: ', s_score[self.s_v==1])
        # print('s_score[self.s_v==0].size(): ', s_score[self.s_v==0].size())
        # print('s_score[self.s_v==0]: ', s_score[self.s_v==0])
        # print('min(s_score[self.s_v==1]): ', min(s_score[self.s_v==1]))
        # print('max(s_score[self.s_v==0]): ', max(s_score[self.s_v==0]))

        # print('t_score[self.t_v==1].size(): ', t_score[self.t_v==1].size())
        # print('t_score[self.t_v==1]: ', t_score[self.t_v==1])
        # print('t_score[self.t_v==0].size(): ', t_score[self.t_v==0].size())
        # print('t_score[self.t_v==0]: ', t_score[self.t_v==0])
        # print('min(t_score[self.t_v==1]): ', min(t_score[self.t_v==1]))
        # print('max(t_score[self.t_v==0]): ', max(t_score[self.t_v==0]))
        return self.s_v, self.t_v

    def generate_new_train_data(self, s_v, t_v, pred_y):
        # create the new dataset merged with pseudo labels
        assert len(t_v) == len(pred_y)
        new_label_flag = list()
        for i, flag in enumerate(t_v):
            if flag > 0:
                new_label_flag.append(pred_y[i])
            elif flag == 0:
                # assign the <unk> pseudo label
                new_label_flag.append(self.args.num_class)
            else:
                raise Exception('Wrong t_v element, legal values are 0 or 1')
        new_label_flag = torch.tensor(new_label_flag)

        # update source data
        if self.args.dataset == 'office':
            new_data = Office_Dataset(root=self.args.data_dir, partition='train', environment=self.args.environment, s_v=s_v, t_v=t_v,
                                       label_flag=new_label_flag, source=self.args.source_name, 
                                       target=self.args.target_name, target_ratio=(self.step+1)*self.args.EF/100, class_num=self.args.source_class_num)
        return new_label_flag, new_data
    
    def get_transferability_score_batch(self, img, label):
        init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(label)

        with torch.no_grad():
                features = self.model(img)
                edge_logits, node_logits = self.gnnModel(init_node_feat=features, init_edge_feat=init_edge, target_mask=target_edge_mask)
                domain_pred = self.discriminator_no_back(features)

                norm_node_logits = F.softmax(node_logits[-1], dim=-1)
                _, target_pred = norm_node_logits.max(-1)
                score = torch.sum(-1*norm_node_logits*torch.log(norm_node_logits), -1)

                # domain_pred.size(): [1, 32] score: [1, 32]
                score = domain_pred - score / torch.log(torch.tensor([self.args.source_class_num])).cuda()
        return score

    def test(self, target_path, step, label_flag):
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        transformer = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean_pix, std=std_pix)])
        target_folder = ImageFolder(target_path, transform=transformer, label_flag=label_flag, return_label_flag=True)
        target_loader = data.DataLoader(target_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

        self.model.eval()
        self.gnnModel.eval()

        per_class_num = np.zeros(self.args.shared_class_num+1)
        per_class_correct = np.zeros(self.args.shared_class_num+1).astype(np.float32)
        class_list = [i for i in range(self.args.shared_class_num)]

        for batch_idx, (img, label, flag) in enumerate(target_loader):
            img, label, flag = img.cuda(), label.cuda(), flag.cuda()
            img = img.unsqueeze(0)
            label = label.squeeze().unsqueeze(0)
            flag = flag.squeeze().unsqueeze(0)
            # print('label: ', label)
            # print('flag: ', flag)
            score = self.get_transferability_score_batch(img, flag)
            init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(flag)
            with torch.no_grad():
                fea = self.model(img)
                edge_logits, node_logits = self.gnnModel(init_node_feat=fea, init_edge_feat=init_edge, target_mask=target_edge_mask)

                norm_node_logits = F.softmax(node_logits[-1], dim=-1)

                if batch_idx == 0:
                    open_class = int(norm_node_logits.size(-1))
                    class_list.append(open_class)
                
                pred = norm_node_logits.data.max(-1)[1]
                ind_unk = np.where(score.squeeze().cpu() < self.args.t_threshold[step])[0]
                ind_unk = torch.tensor(ind_unk).cuda()
                pred[0, ind_unk] = open_class
                pred = pred.cpu().numpy()
                for i, t in enumerate(class_list):
                    t_ind = np.where(label.squeeze().data.cpu().numpy()==t)
                    correct_ind = np.where(pred.squeeze()[t_ind[0]]==t)
                    # print('t_ind: ', t_ind)
                    # print('correct_ind', correct_ind)
                    per_class_correct[i] += float(len(correct_ind[0]))
                    per_class_num[i] += float(len(t_ind[0]))
        print('per_class_correct: ', per_class_correct)
        print('per_class_num: ', per_class_num)
        per_class_acc = per_class_correct / per_class_num
        known_acc = per_class_acc[:len(class_list)-1].mean()
        unknown = per_class_acc[-1]
        h_score = 2 * known_acc * unknown / (known_acc + unknown)
        return h_score, known_acc, unknown
