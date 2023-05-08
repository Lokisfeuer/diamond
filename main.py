# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import numpy as np
import math
import jupyter
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
import os
import pandas as pd
from torchvision.io import read_image
from sklearn.preprocessing import MinMaxScaler


class CustomDiamondDataset(Dataset):
    def __init__(self, data, prices):
        self.x = data
        self.y = prices
        self.length = self.x.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def main():
    data = pd.read_csv('diamonds.csv')
    data, prices, maxi, mini = normalize_data(data)
    # data = data[:100]
    # prices = prices[:100]
    test_input = torch.tensor(np.array([data[2]]), dtype=torch.float32)
    price = get_real_price(prices[2][0], maxi, mini)
    data = torch.tensor(data, dtype=torch.float32)
    prices = torch.tensor(prices, dtype=torch.float32)
    dataset = CustomDiamondDataset(data, prices)
    print(len(data[0]))
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
    model = NeuralNetwork(len(data[0]))
    print(type(data))
    print(type(prices))
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 100

    print(f'Prediction beforehand: {get_real_price(model(test_input).item(), maxi, mini)}\n\ncorrect was: {price}')
    for epoch in range(epochs):
        print(f'Starting new batch {epoch+1}/{epochs}')
        for step, (inputs, labels) in enumerate(dataloader):
            y_pred = model(inputs)
            l = loss(labels, y_pred)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
    print(f'Prediction afterwards: {get_real_price(model(test_input).item(), maxi, mini)}\n\ncorrect was: {price}')
    '''
    n_samples, n_features = data.shape
    input_size = n_features
    model = nn.Linear(input_size, 1) # correct this !
    prices = torch.tensor(prices, dtype=torch.float32)

    loss = nn.MSELoss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(100):
        y_pred = model(X)
        l = loss(Y, y_pred)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
    # output = model(input).item()
    # print(type(data))
    # print(prices)
    # dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
    '''


# both robertas fully copied from https://huggingface.co/sentence-transformers/all-roberta-large-v1
def short_roberta(sentences):
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    embeddings = model.encode(sentences)
    print(embeddings)
    return embeddings


def long_roberta(sentences):
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Sentences we want sentence embeddings for
    # sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
    model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def get_real_price(val, maxi, mini):
    x = (maxi-mini) * val + mini
    return math.exp(x)


def normalize_data(data):
    # don't forget to normalize data - hot encoding
        # max / min normalization the dataset
    # scalars stay normal columns, categories get different columns
    # put price on a logarythmic scale

    '''
    lets build a tensor with the following dimensions:
        carat
        *cut* (hot encoding)
            ideal
            premium
            good
            very good
            fair
        colour (auch hot encoding)
        clarity (dito)
        depth
        table
        price
        x
        y
        z
    then min max the full thing.
    '''
    def onehot():
        nb_classes = 6
        arr = np.array([[2, 3, 4, 0]])
        targets = arr.reshape(-1)
        one_hot_targets = np.eye(nb_classes)[targets]
        return one_hot_targets

    onehot()
    np_data = data.to_numpy()

    cut_index = {'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, 'Ideal':4}
    colour_index = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D':6}
    # clarity: (I1(worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF(best))
    clarity_index = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1':6, 'IF': 7}
    indeces = [cut_index, colour_index, clarity_index]


    # carat, cut (5), colour (7), clarity (8), depth, table, price, x, y
    new_array = []
    prices = []
    for i, diamond in enumerate(np_data):
        diamond = diamond[1:]
        new_diamond = [diamond[0]]
        for j in range(3):
            index = indeces[j][diamond[j+1]]
            zeros = [0.]*len(indeces[j].keys())
            zeros[index] = 1.
            for k in zeros:
                new_diamond.append(k)
        for j in [4, 5, 7, 8]:
            new_diamond.append(diamond[j])
        new_array.append(new_diamond)
        prices.append(math.log(diamond[6]))

    maxi = max(prices)
    mini = min(prices)
    scaler = MinMaxScaler()
    data = pd.DataFrame(new_array)
    prices = pd.DataFrame(prices)
    normalized_data = scaler.fit_transform(data)
    normalized_prices = scaler.fit_transform(prices)

    return normalized_data, normalized_prices, maxi, mini


if __name__ == '__main__':
    main()

