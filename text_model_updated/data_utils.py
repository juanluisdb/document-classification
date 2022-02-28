import logging
import torch
from torch.utils.data import Dataset

import os
import pickle
import re
import numpy as np
from tqdm import trange
from PIL import Image

import h5py




class H5Dataset(Dataset):
    """
    IMDB Dataset for easily iterating over and performing common operations.

    @param (str) path: path of directory where the desired data exists
    @param (pytorch_transformers.BertTokenizer) tokenizer: tokenizer with pre-figured mappings
    @param (bool) apply_cleaning: whether or not to perform common cleaning operations on texts;
           note that enabling only makes sense if language of the task is English
    @param (int) max_tokenization_length: maximum number of positional embeddings, or the sequence
           length of an example that will be fed to BERT model (default: 512)
    @param (str) truncation_method: method that will be applied in case the text exceeds
           @max_tokenization_length; currently implemented methods include 'head-only', 'tail-only',
           and 'head+tail' (default: 'head-only')
    @param (float) split_head_density: weight on head when splitting between head and tail, only
           applicable if @truncation_method='head+tail' (default: 0.5)
    @param (torch.device) device: 'cpu' or 'gpu', decides where to store the data tensors

    """
    def __init__(self, path, tokenizer, phase, use_img=False, img_transforms=None):
        super(H5Dataset).__init__()
        print('dataset init')
        self.file_path = path
        self.dataset = None
        self.img = None
        self.target = None
        self.ocr = None
        self.phase = phase
        self.img_transforms = img_transforms
        self.use_img = use_img
        self.tokenizer = tokenizer

        with h5py.File(self.file_path, 'r') as file:
            if phase == 'train':
                self.dataset_len = len(file["train_ocrs"])
            elif phase == 'val':
                self.dataset_len = len(file["val_ocrs"])
            elif phase == 'test':
                self.dataset_len = len(file["test_ocrs"])


    def pre_tokenize_and_encode_examples(self,example):

        example = re.sub(r'<br />', '', example)
        example = example.lstrip().rstrip()
        example = re.sub(' +', ' ', example)
        
        return self.tokenizer(
            example, 
            padding="max_length", 
            truncation=True,
            return_tensors='pt'
            )


    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        if self.dataset is None:
            if self.phase == 'train':
                self.dataset = h5py.File(self.file_path, 'r')
                #print('File readed')
                if self.use_img:
                    self.img = self.dataset.get('train_img')
                self.target = self.dataset.get('train_labels')
                self.ocr = self.dataset.get('train_ocrs')
            elif self.phase == 'val':
                self.dataset = h5py.File(self.file_path, 'r')
                if self.use_img:
                    self.img = self.dataset.get('val_img')
                self.target = self.dataset.get('val_labels')
                self.ocr = self.dataset.get('val_ocrs')
            elif self.phase == 'test':
                self.dataset = h5py.File(self.file_path, 'r')
                if self.use_img:
                    self.img = self.dataset.get('test_img')
                self.target = self.dataset.get('test_labels')
                self.ocr = self.dataset.get('test_ocrs')

        doc_class = self.target[idx]
        label = torch.tensor(data=doc_class, dtype=torch.long)

        ocr_text = self.ocr[idx]

        if ocr_text == '':
            ocr_text = 'empty'

        example = self.pre_tokenize_and_encode_examples(ocr_text)
        for i in example:
            example[i] = example[i].squeeze()
        example['label'] = label

        if self.use_img:
            img = self.img[idx,:,:,:]
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            img = self.img_transforms(img)
            example['img'] = img

        return example
