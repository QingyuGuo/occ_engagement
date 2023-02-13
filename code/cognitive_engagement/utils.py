import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class CustomizedDataset(Dataset):
    def __init__(self, samples_frame, tokenized_text, contain_label=True):
        '''

        '''
        self.samples_frame = samples_frame
        self.tokenized_text = tokenized_text
        self.contain_label = contain_label

    def __len__(self):
        return len(self.samples_frame)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        input_ids = torch.tensor(self.tokenized_text['input_ids'][index])
        attention_mask = torch.tensor(self.tokenized_text['attention_mask'][index])

        if self.contain_label:
            labels = torch.from_numpy(
                self.samples_frame["label"].values
                ).float().squeeze()
            label = labels[index]

            return (input_ids, attention_mask), label
        else:
            return (input_ids, attention_mask)


def getDataLoader(Batch_Size, tokenizer, shuffle, data_type='train'):

    samples_frame = pd.read_csv('../data/{}.csv'.format(data_type))

    # preprocessed reply; can check details at ./redditcleaner.py
    tokenized_text = tokenizer(
        samples_frame["reply_cleaned"].values.tolist(),
        truncation=True, 
        padding=True
    )

    dataset = CustomizedDataset(
        samples_frame=samples_frame,
        tokenized_text=tokenized_text
        )
    loader = DataLoader(dataset=dataset, batch_size=Batch_Size,
                        shuffle=shuffle, pin_memory=True, num_workers=1)

    return loader


def load_data(Batch_Size, model_name):
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=512
        )

    loader_train = getDataLoader(Batch_Size, tokenizer, True, 'train')

    loader_val = getDataLoader(Batch_Size, tokenizer, False, 'validate')

    loader_test = getDataLoader(Batch_Size, tokenizer, False, 'test')

    return loader_train, loader_val, loader_test
