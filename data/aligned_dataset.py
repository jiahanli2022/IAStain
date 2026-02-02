import os.path
import numpy as np
import torch
import json

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util

import cv2
import numpy as np
import torchvision.transforms as transforms

class AlignedDataset(BaseDataset):
    """
    This dataset class can load aligned/paired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        
        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        if opt.phase == 'train':
            if opt.use_mask:
                self.Mask_paths = sorted(make_dataset(opt.bio_mask_dir, opt.max_dataset_size))
                if opt.use_dino:
                    self.Freq_paths = sorted(make_dataset(os.path.join(opt.dino_mask_dir), opt.max_dataset_size))
                    
        if opt.phase == 'train' and opt.use_scl:
            self.fine_path_label_dictionary = torch.load(opt.fine_path_label_dictionary)
            
            self.fine_label_path_dictionary = torch.load(opt.fine_label_path_dictionary)
            for k, v in self.fine_label_path_dictionary.items():
                
                assert len(v) >= self.opt.sample_per_cluster
            assert f'_clusters{opt.total_cluster_number}_' in opt.fine_path_label_dictionary

           
            self.key_prefix = f'/mnt/nvme1n1/02.processed_data/MIST/BioStainVersion/{opt.dataset_name}-256/TrainValAB/trainA/class_0/'
            self.img_prefix = f'/home/ubuntu/02.data/01.original_data/MIST/{opt.dataset_name}/TrainValAB/trainA/'    # 如果因为跨服务器导致存储的字典和训练图像的名称不匹配，使用img_prefix手动校正
            self.cut_256_image_dir = f"/mnt/nvme1n1/02.processed_data/MIST/BioStainVersion/{opt.dataset_name}-256/TrainValAB/trainA/"


        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        assert self.A_size == self.B_size
        
        self.common_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index = random.randint(0, self.A_size - 1)
            index_B = index % self.B_size
            
        A_path = self.A_paths[index]  # make sure index is within then range
        B_path = self.B_paths[index_B]

        assert A_path == B_path.replace('trainB', 'trainA').replace('valB', 'valA').replace('testB', 'testA')

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if self.opt.phase == 'train' and self.opt.use_mask:
            Mask_path = self.Mask_paths[index]
            assert Mask_path.split('/')[-1] == A_path.split('/')[-1]
            Mask = Image.open(Mask_path).convert('L')

            if self.opt.use_dino:
                freq_path = self.Freq_paths[index]
                assert freq_path.split('/')[-1].replace('.pt', '.jpg') == A_path.split('/')[-1]
                Freq = torch.load(freq_path)


        if not self.opt.use_scl:
            # random crop
            start_x = random.randint(0, np.maximum(0, self.opt.load_size - self.opt.crop_size))
            start_y = random.randint(0, np.maximum(0, self.opt.load_size - self.opt.crop_size))
        else:
            # conditioned random crop
            start_xy_candidates = np.array([
                [0, 0], [0, 256], [0, 512],
                [256, 0], [256, 256], [256, 512],
                [512, 0], [512, 256], [512, 512],
            ])
            crop_xy_candidates = []
            for start_x, start_y in start_xy_candidates:
                flag = True
                for h in [0, 256]:
                    for w in [0, 256]:
                        
                        remap_key = A_path.replace(self.img_prefix, self.key_prefix).replace('.jpg', '_{0:0>6d}_{1:0>6d}.png'.format(start_x + h, start_y + w))
                        if remap_key in self.fine_path_label_dictionary:    # 存在对应的HE patch，才能被训练
                            pass
                        else:
                            flag = False
                if flag == True:
                    crop_xy_candidates.append([start_x, start_y])


            if len(crop_xy_candidates) == 0:
              
                return []

            start_x, start_y = random.choices(crop_xy_candidates)[0] 

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7

        A = transform(A_img)
        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        B = transform(B_img)
            

        A = A[:,start_x: start_x + self.opt.crop_size, start_y: start_y + self.opt.crop_size]   
        B = B[:,start_x: start_x + self.opt.crop_size, start_y: start_y + self.opt.crop_size]
        
        if self.opt.phase == 'train':
            if self.opt.use_mask:
              
                Mask = torch.from_numpy(np.array(Mask))[start_x: start_x + self.opt.crop_size, start_y: start_y + self.opt.crop_size]
                Mask[Mask > 0] = 1

            
                if self.opt.use_dino:
                    # 这里的start_y和start_x顺序要特别小心，不过以下这一句已经经过验证
                    Freq = Freq[start_x: start_x + self.opt.crop_size, start_y: start_y + self.opt.crop_size]


            if self.opt.use_scl:
                SCL_Pool, Cluster_Number = [], []
                for h in [0, 256]:
                    for w in [0, 256]:
                        remap_key = A_path.replace(self.img_prefix, self.key_prefix).replace('.jpg', '_{0:0>6d}_{1:0>6d}.png'.format(start_x + h, start_y + w))
                        cluster_number = self.fine_path_label_dictionary[remap_key]
                        Cluster_Number.append(cluster_number)

                        sample_candidates = self.fine_label_path_dictionary[cluster_number]
                        K = self.opt.sample_per_cluster
                        sampled_imgs = np.random.choice(sample_candidates, size=K, replace=False).tolist()

                        SCL_Pool.extend([
                            self.common_transform(Image.open(p.replace(self.key_prefix,self.cut_256_image_dir).replace('class_0/','').replace('trainA', 'trainB').replace('valA', 'valB')).convert('RGB')) for p in sampled_imgs
                        ])

                SCL_Pool = torch.stack(SCL_Pool, 0).unsqueeze(0)
                

                additional_SCL_NegPool = []

                other_Cluster_Number = set(range(self.opt.total_cluster_number)).difference(set(Cluster_Number))
                sampled_other_Cluster_Numbers = np.random.choice(list(other_Cluster_Number), size=self.opt.other_cluster_number, replace=False).tolist()

                for cluster_number in sampled_other_Cluster_Numbers:
                    sample_candidates = self.fine_label_path_dictionary[cluster_number]
                    K = self.opt.sample_per_cluster
                    sampled_imgs = np.random.choice(sample_candidates, size=K, replace=False).tolist()
                    additional_SCL_NegPool.extend([
                            self.common_transform(Image.open(p.replace(self.key_prefix,self.cut_256_image_dir).replace('class_0/','').replace('trainA', 'trainB').replace('valA', 'valB')).convert('RGB')) for p in sampled_imgs
                        ])
                additional_SCL_NegPool = torch.stack(additional_SCL_NegPool, 0).unsqueeze(0)    


                Cluster_Number = torch.Tensor(Cluster_Number).unsqueeze(0)     

        if self.opt.phase == 'train' and self.opt.use_mask:
            if self.opt.use_dino:
                if self.opt.use_scl:
                    return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'Mask':Mask, 'Freq': Freq, 'SCL_Pool': SCL_Pool, 'additional_SCL_NegPool': additional_SCL_NegPool, 'Cluster_Number': Cluster_Number}
                else:
                    return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'Mask':Mask, 'Freq': Freq}
            else:
                return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'Mask':Mask}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)