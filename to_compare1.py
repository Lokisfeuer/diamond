# TODO comment to everything its purpose

'''
# These are the start command when running this as jupyter notebook on colabs:

print('starting.')
!pip install transformers
!pip install sentence_transformers
!pip install torchmetrics
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
'''

import math

# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from torchmetrics import R2Score


class CustomMovieDataset(Dataset):
    def __init__(self, reviews, sentiments):
        self.x = reviews
        self.y = sentiments
        self.length = self.x.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits



def main(epochs=10, learning_rate=0.01, train_batch_size=10, loss=None):
    if loss is None:
        loss = nn.MSELoss()  # TODO pass loss as function object
    url = 'https://raw.githubusercontent.com/Lokisfeuer/diamond/master/imdbdataset.csv'
    data = pd.read_csv(url)
    data, sentiments = prepare_data(data)

    dataset = CustomMovieDataset(data, sentiments)

    dataloader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)
    print(data)
    model = NeuralNetwork(len(data[0]))
    r2loss = R2Score()
    mseloss = nn.MSELoss()
    bceloss = nn.BCELoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_l = 0
        print(f'Starting new batch {epoch + 1}/{epochs}')
        for step, (inputs, labels) in enumerate(dataloader):
            y_pred = model(inputs)
            # l = loss(y_pred, labels)
            l = loss(y_pred, labels)
            running_l += l.item()
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (step + 1) % 50 == 0:  # if (step+1) % 100 == 0:
                print(f'training loss: {running_l / 50}')
                running_l = 0


def prepare_data(data):
    np_data = data.to_numpy().transpose()
    file_data = torch.load('embedded_reviews.pt')
    print(type(file_data))
    x = []
    print(f'length of file_data: {len(file_data)}')
    for i in file_data:
        x.append(torch.from_numpy(i))
    print(f'length of x: {len(x)}')
    reviews = torch.cat(x)
    # reviews = model.encode(np_data[0])
    sentiments = np_data[1][:7100]
    print(f'length of reviews: {len(reviews)}')
    print(f'length of sentiments: {len(sentiments)}')
    sentiments[sentiments == 'positive'] = [1.]
    sentiments[sentiments == 'negative'] = [0.]
    sents = []
    for i in sentiments:
        sents.append([i])
    sentiments = np.array(sents, dtype=np.float32)
    sentiments = torch.from_numpy(sentiments)
    reviews = torch.tensor(reviews, dtype=torch.float32)
    sentiments = torch.tensor(sentiments, dtype=torch.float32)  # line needed? dtype?
    return reviews, sentiments


if __name__ == '__main__':
    # prepare_data_slowly()
    kwargs = {
        'epochs':10,
        'learning_rate':0.01,
        'train_batch_size':25,
        'loss':nn.BCEWithLogitsLoss()
    }
    main(**kwargs)
    # for jupyter:
    #   change reading of csv
    #   adjust start command from jupyter.
