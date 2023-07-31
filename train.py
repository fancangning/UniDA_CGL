import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import time
# torch-related packages
import torch
# data-related packages
from data_loader import Office_Dataset
from model_trainer import ModelTrainer

# torch.cuda.set_device(0)


def main(args):
    total_step = 100 // args.EF
    # oda office31 A-W
    # args.s_threshold = [-1] * total_step
    # args.t_threshold = [-0.87, -0.65, -0.75, -0.75, -0.5, -0.75, -0.75, -0.75, -0.75, -0.5, -0.6, -0.5, -0.5, -0.5, -0.5, -0.5, -0.25, -0.35, -0.25, -0.35]
    
    # opda office31 A-W
    args.s_threshold = [-0.2, -0.35, -0.5, -0.45, -0.45, -0.4, -0.4, -0.4, -0.3, 0.1, 0, 0, 0.1, 0.25, 0.25, 0.25, 0.15, 0.2, 0.4, 0.75]
    args.t_threshold = [-0.75, -0.75, -0.8, -0.75, -0.75, -0.75, -0.6, -0.6, -0.6, -0.55, -0.7, -0.6, -0.6, -0.55, -0.55, -0.55, -0.5, -0.4, -0.4, -0.4]

    # set the log file
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    args.log_file = open(os.path.join(args.logs_dir, args.environment+'_'+args.dataset+'_'+args.source_name+'-'+args.target_name+'_log.txt'), 'w')
    args.log_file.write(str(vars(args))+'\n')
    args.log_file.flush()

    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # prepare checkpoints and log folders
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    
    # initialize dataset
    if args.dataset == 'office':
        args.data_dir = os.path.join(args.data_dir, 'office31')
        data = Office_Dataset(root=args.data_dir, partition='train', environment=args.environment, s_v=None, t_v=None, label_flag=None, source=args.source_name, 
                              target=args.target_name, class_num=args.source_class_num)
    else:
        print('Unknown dataset!')

    args.num_class = data.num_class
    args.alpha = data.alpha

    # setting experiment name
    label_flag = None
    s_selected_idx = None
    t_selected_idx = None
    # args.experiment is a str
    args.experiment = set_exp_name(args)

    if not args.visualization:
        # total_step = 100 // args.EF
        total_time = 0
        for step in range(total_step):
            print('This is {}-th step with EF={}%'.format(step, args.EF))
            args.log_file.write('This is {}-th step with EF={}%'.format(step, args.EF))
            args.log_file.write('\n')
            args.log_file.flush()
            trainer = ModelTrainer(args=args, data=data, step=step, label_flag=label_flag, 
                                   s_v=s_selected_idx, t_v=t_selected_idx, logger=None)

            # train the model
            # args.log_epoch 4 4 5 5 6 6 7 7 8 8
            args.log_epoch = 4 + step // 2
            # epochs 4 6 8 10 12 14 16 18 20 22 24
            start = time.time()
            trainer.train(step, epochs=4+(step)*2, step_size=args.log_epoch)
            end = time.time()
            args.log_file.write('current step cost '+str(end-start)+' seconds')
            args.log_file.write('\n')
            args.log_file.flush()
            total_time += end - start

            # test the model
            h_score, known_acc, unknown_acc = trainer.test(data.target_path, step, data.label_flag)

            print('The '+str(step)+' step of total '+str(total_step-1)+' step, h_score: '+str(h_score)+' known_acc: '+str(known_acc)+' unknown_acc: '+str(unknown_acc))
            args.log_file.write('The '+str(step)+' step of total '+str(total_step-1)+' step, h_score: '+str(h_score)+' known_acc: '+str(known_acc)+' unknown_acc: '+str(unknown_acc))
            args.log_file.write('\n')
            args.log_file.flush()

            # transferability score
            s_score, t_score, pred_y = data.transferability_score(trainer.model, trainer.gnnModel, trainer.discriminator_no_back)

            # select transferable source and target data
            s_selected_idx, t_selected_idx = trainer.select_top_data(s_score, t_score, step)
            print('the amount of source samples: ', len(s_selected_idx))
            print('the amount of target samples: ', len(t_selected_idx))
            print('the amount of source selected samples: ', sum(s_selected_idx))
            print('the amount of target selected samples: ', sum(t_selected_idx))

            # add new data
            label_flag, data = trainer.generate_new_train_data(s_selected_idx, t_selected_idx, pred_y)
        args.log_file.write('average cost '+str(total_time/total_step)+' seconds')
        args.log_file.write('\n')
        args.log_file.flush()
        args.log_file.close()
    else:
        # load trained weights
        raise Exception('visualization has not been completed')


def set_exp_name(args):
    exp_name = 'D-{}'.format(args.dataset)
    if args.dataset == 'office':
        exp_name += '_src-{}_tar-{}'.format(args.source_name, args.target_name)
    exp_name += '_A-{}'.format(args.arch)
    exp_name += '_L-{}'.format(args.num_layers)
    exp_name += '_E-{}_B-{}'.format(args.EF, args.batch_size)
    return exp_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curriculum Graph Learning for Universal Domain Adaptation')
    
    # set up dataset & backbone embedding
    parser.add_argument('--gpu_devices', type=int, default=0, help='which gpu to use')
    parser.add_argument('--dataset', type=str, default='office')
    parser.add_argument('-a', '--arch', type=str, default='res')
    parser.add_argument('--root_path', type=str, default='./utils/', metavar='B', help='root dir')

    # set up path
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'txt'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'logs_opda'))
    parser.add_argument('--checkpoints_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'checkpoints_opda'))

    # verbose setting
    parser.add_argument('--log_step', type=int, default=30)
    parser.add_argument('--log_epoch', type=int, default=3)

    parser.add_argument('--environment', type=str, default='opda')
    parser.add_argument('--source_name', type=str, default='A')
    parser.add_argument('--target_name', type=str, default='W')

    parser.add_argument('--source_class_num', type=int, default=20, help='the number of source classes')
    parser.add_argument('--shared_class_num', type=int, default=10, help='the number of shared classes')

    parser.add_argument('--eval_log_step', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=1500)

    # hyper-parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('--s_threshold', type=float, default=-1)
    parser.add_argument('--t_threshold', type=float, default=-1)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--EF', type=int, default=5)
    parser.add_argument('--loss', type=str, default='focal')


    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    # GNN parameters
    parser.add_argument('--in_features', type=int, default=2048)
    # if dataset == 'home':
    #     parser.add_argument('--node_features', type=int, default=512)
    #     parser.add_argument('--edge_features', type=int, default=512)
    # else:
    #     parser.add_argument('--node_features', type=int, default=1024)
    #     parser.add_argument('--edge_features', type=int, default=1024)
    parser.add_argument('--node_features', type=int, default=1024)
    parser.add_argument('--edge_features', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=1)

    #tsne
    parser.add_argument('--visualization', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/D-visda18_A-res_L-1_E-20_B-4_step_1.pth.tar')

    #Discrminator
    parser.add_argument('--discriminator', type=bool, default=True)
    parser.add_argument('--adv_coeff', type=float, default=0.4)

    #GNN hyper-parameters
    parser.add_argument('--node_loss', type=float, default=0.3)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_devices)
    main(args)
