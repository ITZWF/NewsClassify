# encoding: utf-8
import os
from torch.utils.data import Dataset


class ClsDataset(Dataset):

    def __init__(self, dirs, cls_dict, fre_dict, vocab_size):
        self.dirs = dirs
        self.cls_dict = cls_dict
        self.fre_dict = fre_dict
        self.vocab_size: int = vocab_size
        self.name_list = list()
        for clss in os.listdir(self.dirs):
            clss_path = self.dirs + '/' + clss
            for file in os.listdir(clss_path):
                new_path = clss_path + '/' + file
                if new_path not in self.name_list:
                    self.name_list.append(new_path)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        label = name.split('/')[-2]
        sample = {'label': self.cls_dict['forward'][label], 'text': []}
        with open(name, mode='r', encoding='utf-8') as f:
            words = f.read().split(' ')
            for word in words:
                if word in self.fre_dict:
                    sample['text'].append(word)
                else:
                    sample['text'].append(self.fre_dict['<UNK>'])
        if len(sample['text']) >= self.vocab_size:
            sample['text'] = sample['text'][:self.vocab_size]
        else:
            for i in range(len(sample['text']), self.vocab_size):
                sample['text'].append(self.fre_dict['<PAD>'])
        return sample
