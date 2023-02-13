# -*- coding: utf-8 -*-


from cProfile import label
import re
import torch
import pandas as pd
import numpy as np
import liwc
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn import preprocessing
from nltk.tokenize import sent_tokenize, word_tokenize
from emoji import emoji_count
import pickle
from pandarallel import pandarallel
import redditcleaner

pandarallel.initialize(nb_workers=2, progress_bar=True)


class CustomizedDataset(Dataset):
    def __init__(
        self,
        tokenized_content,
        content_stat_vecs,
        liwc_count_vecs,
        labels
    ):

        self.tokenized_content = tokenized_content
        self.content_stat_vecs = content_stat_vecs
        self.liwc_count_vecs = liwc_count_vecs
        self.labels = "no_label"
        self.labels = labels

    def __len__(self):
        return len(self.content_stat_vecs)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_ids_content = torch.tensor(
            self.tokenized_content['input_ids'][idx])
        attention_mask_content = torch.tensor(
            self.tokenized_content['attention_mask'][idx])

        content_stat_vecs = torch.from_numpy(
            self.content_stat_vecs).float().squeeze()
        content_stat_vec = content_stat_vecs[idx]

        liwc_count_vecs = torch.tensor(
            self.liwc_count_vecs).float().squeeze()
        liwc_count_vec = liwc_count_vecs[idx]

        if hasattr(self, 'labels'):
            labels = torch.from_numpy(
                self.labels).float().squeeze()
            label = labels[idx]
            sample = {
                "content_cleaned": (input_ids_content, attention_mask_content),
                "content_stat_vec": content_stat_vec,
                "liwc_count_vec": liwc_count_vec,
                "label": label
            }
            return sample
        else:
            sample = {
                "content_cleaned": (input_ids_content, attention_mask_content),
                "content_stat_vec": content_stat_vec,
                "liwc_count_vec": liwc_count_vec
            }
            return sample


def save_pkl(objs, path):
    with open(path, 'wb') as f:
        pickle.dump(objs, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)



def cal_digit(str):
    digit = 0
    for ch in str:
        if ch.isdigit():
            digit = digit+1
        else:
            pass
    return digit


def get_content_stat_vec(obj):
    word_num = len(sent_tokenize(obj))
    sentence_num = len(word_tokenize(obj))
    char_num = len(obj)
    url_num = obj.count('[URL]')
    emoji_number = emoji_count(obj)
    digit_number = cal_digit(obj)
    vec = [word_num, sentence_num, char_num,
           url_num, emoji_number, digit_number]
    return vec


def liwc_tokenize(text):
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


def get_liwc_count_vec(text):
    parse, category_names = liwc.load_token_parser('../liwc/LIWC2015_English.dic')

    counter = None
    if pd.isna(text) or len(text) == 0:
        counter = Counter()
    else:
        text = liwc_tokenize(text.lower())
        counter = Counter(category for token in text
                          for category in parse(token))

    return [counter.get(category, 0) for category in category_names]


def get_standard_scaler():
    '''
    learn data distribution based on the training and validation set.
    '''

    df_train = pd.read_csv('train.csv')
    df_val = pd.read_csv('validate.csv')
    df = pd.concat([df_train, df_val]).reset_index()

    content_stat_vecs = np.array(df['content_cleaned'].parallel_apply(
        get_content_stat_vec).tolist())
    liwc_count_vecs = np.array(df['content_cleaned'].parallel_apply(
        get_liwc_count_vec).tolist())
    scaler_content = preprocessing.StandardScaler().fit(content_stat_vecs)
    scaler_liwc = preprocessing.StandardScaler().fit(liwc_count_vecs)



    return scaler_content, scaler_liwc



def getDataLoader(Batch_Size, tokenizer, scaler_content, scaler_liwc, shuffle, data_type='train'):

    samples_frame = pd.read_csv('{}.csv'.format(data_type))


    # content_cleaned -- we concatenate the title and post content (selftext); replace the URL with [URL]; and apply the redditcleaner;
    samples_frame["content_cleaned"].fillna("")

    # content embedding -- from language model
    tokenized_content = tokenizer(
        samples_frame["content_cleaned"].astype(str).values.tolist(),
        truncation=True,
        padding=True
    )

    # statistic features, e.g., URL number
    content_stat_vecs = np.array(samples_frame['content_cleaned'].apply(
        get_content_stat_vec).tolist())
    # standardized features
    content_stat_vecs_scaled = scaler_content.transform(content_stat_vecs)

    # LIWC features
    liwc_count_vecs = np.array(samples_frame['content_cleaned'].apply(
        get_liwc_count_vec).tolist())
    # standardized features
    liwc_count_vecs_scaled = scaler_liwc.transform(liwc_count_vecs)

    labels = samples_frame["label"].values

    dataset = CustomizedDataset(
        tokenized_content=tokenized_content,
        content_stat_vecs=content_stat_vecs_scaled,
        liwc_count_vecs=liwc_count_vecs_scaled,
        labels=labels
    )
    loader = DataLoader(dataset=dataset, batch_size=Batch_Size,
                        shuffle=shuffle, pin_memory=True, num_workers=1)

    return loader


def load_data(Batch_Size, model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=512
    )

    
    scaler_content, scaler_liwc = get_standard_scaler()

    loader_train = getDataLoader(
        Batch_Size, tokenizer, scaler_content, scaler_liwc, True, 'train')

    loader_val = getDataLoader(
        Batch_Size, tokenizer, scaler_content, scaler_liwc, False, 'validate')

    loader_test = getDataLoader(
        Batch_Size, tokenizer, scaler_content, scaler_liwc, False, 'test')

    return loader_train, loader_val, loader_test

