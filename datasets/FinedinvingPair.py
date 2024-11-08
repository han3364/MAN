import torch
import numpy as np
import os
import pickle
import random
import glob
from os.path import join
from PIL import Image
from opts import *
from scipy import stats
from torchvideotransforms import video_transforms, volume_transforms
from utils.util import Group_helper

class FineDiving_Pair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform
        self.random_choosing = args.random_choosing
        self.action_number_choosing = args.action_number_choosing
        self.length = args.frame_length
        self.voter_number = args.voter_number

        # file path
        self.data_root = args.data_root
        self.data_anno = self.read_pickle(args.label_path)
        with open(args.train_split, 'rb') as f:
            self.train_dataset_list = pickle.load(f)
        with open(args.test_split, 'rb') as f:
            self.test_dataset_list = pickle.load(f)

        self.action_number_dict = {}
        self.difficulties_dict = {}
        if self.subset == 'train':
            self.dataset = self.train_dataset_list
        else:
            self.dataset = self.test_dataset_list
            self.action_number_dict_test = {}
            self.difficulties_dict_test = {}

        self.choose_list = self.train_dataset_list.copy()
        if self.action_number_choosing:
            self.preprocess()
            self.check_exemplar_dict()

    def preprocess(self):
        for item in self.train_dataset_list:
            dive_number = self.data_anno.get(item)[0]
            if self.action_number_dict.get(dive_number) is None:
                self.action_number_dict[dive_number] = []
            self.action_number_dict[dive_number].append(item)
        if self.subset == 'test':
            for item in self.test_dataset_list:
                dive_number = self.data_anno.get(item)[0]
                if self.action_number_dict_test.get(dive_number) is None:
                    self.action_number_dict_test[dive_number] = []
                self.action_number_dict_test[dive_number].append(item)
  
    def delta(self):
        '''
            RT: builder group
        '''
        delta = []
        for key in list(self.action_number_dict.keys()):
            file_list = self.action_number_dict[key]
            for i in range(len(file_list)):
                for j in range(i + 1,len(file_list)):
                    delta.append(abs(
                        self.data_anno[file_list[i]][1] / self.data_anno[file_list[i]][2] - 
                        self.data_anno[file_list[j]][1] / self.data_anno[file_list[j]][2]))
                    
        return delta

    def check_exemplar_dict(self):
        if self.subset == 'train':
            for key in sorted(list(self.action_number_dict.keys())):
                file_list = self.action_number_dict[key]
                for item in file_list:
                    assert self.data_anno[item][0] == key
        if self.subset == 'test':
            for key in sorted(list(self.action_number_dict_test.keys())):
                file_list = self.action_number_dict_test[key]
                for item in file_list:
                    assert self.data_anno[item][0] == key

    def load_video(self, video_file_name):
        image_list = sorted((glob.glob(os.path.join(self.data_root, video_file_name[0], str(video_file_name[1]), '*.jpg'))))

        start_frame = int(image_list[0].split("\\")[-1][:-4])
        end_frame = int(image_list[-1].split("\\")[-1][:-4])
        frame_list = np.linspace(start_frame, end_frame, self.length).astype(int)
        image_frame_idx = [frame_list[i] - start_frame for i in range(self.length)]

        video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.length)]
        frames_labels = [self.data_anno.get(video_file_name)[4][i] for i in image_frame_idx]
        frames_catogeries = list(set(frames_labels))
        frames_catogeries.sort(key=frames_labels.index)
        transitions = [frames_labels.index(c) for c in frames_catogeries]
        return self.transforms(video), np.array([transitions[1]-1,transitions[-1]-1]), np.array(frames_labels)


    def read_pickle(self, pickle_path):
        with open(pickle_path,'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __getitem__(self, index):
        sample_1  = self.dataset[index]
        data = {}
        data['video'], data['transits'], data['frame_labels'] = self.load_video(sample_1)
        data['number'] = self.data_anno.get(sample_1)[0]
        data['final_score'] = self.data_anno.get(sample_1)[1]
        data['difficulty'] = self.data_anno.get(sample_1)[2]
        data['completeness'] = (data['final_score'] / data['difficulty'])
        # self.proc_label(data)

        # choose a exemplar
        if self.subset == 'train':
            # train phrase
            if self.action_number_choosing == True:
                file_list = self.action_number_dict[self.data_anno[sample_1][0]].copy()
            elif self.DD_choosing == True:
                file_list = self.difficulties_dict[self.data_anno[sample_1][2]].copy()
            else:
                # randomly
                file_list = self.train_dataset_list.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            target['video'], target['transits'], target['frame_labels'] = self.load_video(sample_2)
            target['number'] = self.data_anno.get(sample_2)[0]
            target['final_score'] = self.data_anno.get(sample_2)[1]
            target['difficulty'] = self.data_anno.get(sample_2)[2]
            target['completeness'] = (target['final_score'] / target['difficulty'])
            return data, target
        else:
            # test phrase
            if self.action_number_choosing:
                train_file_list = self.action_number_dict[self.data_anno[sample_1][0]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            elif self.DD_choosing:
                train_file_list = self.difficulties_dict[self.data_anno[sample_1][2]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            else:
                # randomly
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['video'], tmp['transits'], tmp['frame_labels'] = self.load_video(item)
                tmp['number'] = self.data_anno.get(item)[0]
                tmp['final_score'] = self.data_anno.get(item)[1]
                tmp['difficulty'] = self.data_anno.get(item)[2]
                tmp['completeness'] = (tmp['final_score'] / tmp['difficulty'])
                target_list.append(tmp)
            return data, target_list

    def __len__(self):
        return len(self.dataset)

def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((455,256)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((455,256)),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_FDPair_dataloader(args):
    train_trans, test_trans = get_video_trans()
    train_dataset = FineDiving_Pair_Dataset(args, 'train', train_trans)
    test_dataset = FineDiving_Pair_Dataset(args, 'test', test_trans)

    grout_helper = Group_helper(train_dataset.delta(), depth=5, Max=30, Min=0)

    train_sampler = None
    test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                               shuffle=True,
                                               num_workers=int(args.num_workers),
                                               pin_memory=True, sampler=train_sampler,
                                               drop_last=True,
                                               worker_init_fn=worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              num_workers=int(args.num_workers),
                                              pin_memory=True,
                                              shuffle=False,
                                              drop_last=True,
                                              sampler=test_sampler
                                              )
    return train_dataloader, test_dataloader, grout_helper