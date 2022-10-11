#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
from typing import List

import sys

import tokenizers
import torch
from torch.utils.data import Dataset, DataLoader
from pypinyin import pinyin, Style
from transformers import BertTokenizer
from datasets.chinese_bert_dataset import ChineseBertDataset

from datasets.utils import pho_convertor
import pickle


class Dynaimic_CSCDataset(ChineseBertDataset):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        file = self.data_path
        print('processing ',file)
        with open(file, 'r' ,encoding='utf8') as f:
            self.data = list(f.readlines())
        self.data = [line for line in self.data if len(json.loads(line)['input_ids']) < 192]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = json.loads(self.data[idx])
        input_ids, pinyin_ids, label,pinyin_label = example['input_ids'], example['pinyin_ids'], example['label'], example['pinyin_label']
        tgt_pinyin_ids = example['tgt_pinyin_ids']
        # convert list to tensor
        input_ids = torch.LongTensor(input_ids)
        pinyin_ids = torch.LongTensor(pinyin_ids).view(-1)

        label = torch.LongTensor(label)
        pinyin_label=torch.LongTensor(pinyin_label)
        tgt_pinyin_ids = torch.LongTensor(tgt_pinyin_ids).view(-1)
        return input_ids, pinyin_ids, label, tgt_pinyin_ids, pinyin_label



class TestCSCDataset(ChineseBertDataset):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # convert sentence to ids
        sentence=self.data[idx]['src']
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # assert
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        sentence=self.data[idx]['tgt']
        tokenizer_output = self.tokenizer.encode(sentence)

        assert len(bert_tokens) == len(tokenizer_output.ids)
        label = torch.LongTensor(tokenizer_output.ids)
        pinyin_label=torch.LongTensor(self.convert_sentence_to_shengmu_yunmu_shengdiao_ids(sentence, tokenizer_output))
        example_id=self.data[idx]['id']
        src=self.data[idx]['src']
        tokens_size=self.data[idx]['tokens_size']
        return input_ids, pinyin_ids, label,pinyin_label,example_id,src,tokens_size


