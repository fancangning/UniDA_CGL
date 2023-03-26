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

class Office_Dataset(Base_Dataset):

    def __init__(self, root, partition, s_v=None, t_v=None, label_flag=None, source='A', target='W', target_ratio=0.0):
        super(Office_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join(root, src_name)
        self.target_path = os.path.join(root, tar_name)
        self.class_name = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
                           'laptop_computer', 'monitor', 'mouse', 'mug', 'projector', 'unk']
        self.num_class = len(self.class_name)
        self.s_v = s_v
        # self.source_image is a dict, and self.target_image and self.target_label are lists.
        self.source_image, self.s_v, self.target_image, self.target_label = self.load_dataset()
        self.alpha = [len(self.source_image(key)) for key in self.source_image.keys()]
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
    
    def getFilePath(self, source, target):

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
        
        return src_name, tar_name
        