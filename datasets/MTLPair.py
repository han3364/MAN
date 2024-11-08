import torch
import numpy as np
import os
import pickle
import random
import glob
from utils.util import Group_helper
from torchvideotransforms import video_transforms, volume_transforms
# from os.path import join
from PIL import Image

class MTLPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(0)
        self.subset = subset
        self.transforms = transform
        # using Difficult Degree
        self.usingDD = True
        # some flags
        self.dive_number_choosing = False
        # file path
        self.label_path = './data/info/augmented_final_annotations_dict.pkl'
        self.split_path = './data/info/train_split_0.pkl'
        self.split = self.read_pickle(self.split_path)
        self.label_dict = self.read_pickle(self.label_path)
        self.data_root = 'C:/datasets/MTL/frames'
        # setting
        self.temporal_shift = [-3, 3]
        self.voter_number = 10
        self.frame_length = 103
        # build difficulty dict ( difficulty of each action, the cue to choose exemplar)
        self.difficulties_dict = {}
        self.dive_number_dict = {}
        if self.subset == 'test':
            self.split_path_test = './data/info/test_split_0.pkl'
            self.split_test = self.read_pickle(self.split_path_test)
            self.difficulties_dict_test = {}
            self.dive_number_dict_test = {}
        if self.usingDD:
            self.preprocess()
            self.check()
                
        self.choose_list = self.split.copy()
        if self.subset == 'test':
            self.dataset = self.split_test
        else:
            self.dataset = self.split
 
    def load_video(self, key, phase):
        image_list = sorted((glob.glob(os.path.join(self.data_root,
                                                    str('{:02d}_{:02d}'.format(key[0], key[1])),
                                                    '*.jpg'))))
        
        frame_start_idx = 0
        if phase == 'train':
            temporal_aug_shift = random.randint(self.temporal_shift[0], self.temporal_shift[1])
            frame_start_idx = 3 + temporal_aug_shift

        video = [Image.open(image_path) for image_path in image_list[frame_start_idx:frame_start_idx+self.frame_length]]
        return self.transforms(video)


    def read_pickle(self,pickle_path):
        with open(pickle_path,'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data


    def preprocess(self):
        if self.dive_number_choosing:
            # Dive Number
            for item in self.split:
                dive_number = self.label_dict.get(item)['dive_number']
                if self.dive_number_dict.get(dive_number) is None:
                    self.dive_number_dict[dive_number] = []
                self.dive_number_dict[dive_number].append(item)

            if self.subset == 'test':
                for item in self.split_test:
                    dive_number = self.label_dict.get(item)['dive_number']
                    if self.dive_number_dict_test.get(dive_number) is None:
                        self.dive_number_dict_test[dive_number] = []
                    self.dive_number_dict_test[dive_number].append(item)
        else:        
            # DD
            for item in self.split:
                difficulty = self.label_dict.get(item)['difficulty']
                if self.difficulties_dict.get(difficulty) is None:
                    self.difficulties_dict[difficulty] = []
                self.difficulties_dict[difficulty].append(item)

            if self.subset == 'test':
                for item in self.split_test:
                    difficulty = self.label_dict.get(item)['difficulty']
                    if self.difficulties_dict_test.get(difficulty) is None:
                        self.difficulties_dict_test[difficulty] = []
                    self.difficulties_dict_test[difficulty].append(item)
            


    def check(self):
        if self.dive_number_choosing:
            # dive_number_dict
            for key in sorted(list(self.dive_number_dict.keys())):
                file_list = self.dive_number_dict[key]
                for item in file_list:
                    assert self.label_dict[item]['dive_number'] == key

            if self.subset == 'test':
                for key in sorted(list(self.dive_number_dict_test.keys())):
                    file_list = self.dive_number_dict_test[key]
                    for item in file_list:
                        assert self.label_dict[item]['dive_number'] == key
        else:
            # difficulties_dict
            for key in sorted(list(self.difficulties_dict.keys())):
                file_list = self.difficulties_dict[key]
                for item in file_list:
                    assert self.label_dict[item]['difficulty'] == key

            if self.subset == 'test':
                for key in sorted(list(self.difficulties_dict_test.keys())):
                    file_list = self.difficulties_dict_test[key]
                    for item in file_list:
                        assert self.label_dict[item]['difficulty'] == key

        print('check done')

    def delta(self):
        '''
            RT: builder group
        '''
        if self.usingDD:
            if self.dive_number_choosing:
                delta = []
                for key in list(self.dive_number_dict.keys()):
                    file_list = self.dive_number_dict[key]
                    for i in range(len(file_list)):
                        for j in range(i + 1,len(file_list)):
                            delta.append(abs(
                                self.label_dict[file_list[i]]['final_score'] / self.label_dict[file_list[i]]['difficulty'] - 
                                self.label_dict[file_list[j]]['final_score'] / self.label_dict[file_list[j]]['difficulty']))
            else:
                delta = []
                for key in list(self.difficulties_dict.keys()):
                    file_list = self.difficulties_dict[key]
                    for i in range(len(file_list)):
                        for j in range(i + 1,len(file_list)):
                            delta.append(abs(
                                self.label_dict[file_list[i]]['final_score'] / self.label_dict[file_list[i]]['difficulty'] - 
                                self.label_dict[file_list[j]]['final_score'] / self.label_dict[file_list[j]]['difficulty']))
        else:
            delta = []
            dataset = self.split.copy()
            for i in range(len(dataset)):
                for j in range(i+1,len(dataset)):
                    delta.append(
                        abs(
                            self.label_dict[dataset[i]]['final_score'] -
                            self.label_dict[dataset[j]]['final_score']))

        return delta

    def __getitem__(self,index):
        sample_1  = self.dataset[index]
        data = {}
        if self.subset == 'test':
            # test phase
            data['video'] = self.load_video(sample_1, 'test')
            data['final_score'] = self.label_dict.get(sample_1).get('final_score')
            data['difficulty'] = self.label_dict.get(sample_1).get('difficulty')
            data['completeness'] = (data['final_score'] /  data['difficulty'] )

            if self.usingDD:
                # NOTE: using Dive Number to choose
                if  self.dive_number_choosing:
                    train_file_list = self.dive_number_dict[self.label_dict[sample_1]['dive_number']]
                    random.shuffle(train_file_list)
                    choosen_sample_list = train_file_list[:self.voter_number] 
                else:
                    # choose a list of sample in training_set
                    train_file_list = self.difficulties_dict[self.label_dict[sample_1]['difficulty']]
                    random.shuffle(train_file_list)
                    choosen_sample_list = train_file_list[:self.voter_number]
            else:
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]

            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['video'] = self.load_video(item, 'test')  
                tmp['final_score'] = self.label_dict.get(item).get('final_score')
                tmp['difficulty'] = self.label_dict.get(item).get('difficulty') 
                tmp['completeness'] = (tmp['final_score'] /  tmp['difficulty']) 
                # print(tmp)
                target_list.append(tmp)
                
            return data , target_list
        else:
            # train phase
            data['video'] = self.load_video(sample_1, 'train')
            data['final_score'] = self.label_dict.get(sample_1).get('final_score')
            data['difficulty'] = self.label_dict.get(sample_1).get('difficulty') 
            data['completeness'] = (data['final_score'] /  data['difficulty'] )

            # choose a sample
            if self.usingDD:
            # did not using a pytorch sampler, using diff_dict to pick a video sample                
                if  self.dive_number_choosing:
                    # NOTE: using Dive Number to choose
                    file_list = self.dive_number_dict[self.label_dict[sample_1]['dive_number']].copy()
                else:
                    # all sample owning same difficulties
                    file_list = self.difficulties_dict[self.label_dict[sample_1]['difficulty']].copy()
            else:
                # randomly
                file_list = self.split.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            # sample 2
            target['video'] = self.load_video(sample_2, 'train')
            target['final_score'] = self.label_dict.get(sample_2).get('final_score')
            target['difficulty'] = self.label_dict.get(sample_2).get('difficulty') 
            target['completeness'] = (target['final_score'] /  target['difficulty']) 
            return data , target
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

def get_MTLPair_dataloader(args):
    train_trans, test_trans = get_video_trans()
    train_dataset = MTLPair_Dataset(args, 'train', train_trans)
    test_dataset = MTLPair_Dataset(args, 'test', test_trans)

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
