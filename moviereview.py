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
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class History():
    def __init__(self, val_set, train_set, model, **kwargs):
        self.val_set = val_set
        self.train_set = train_set
        self.model = model
        self.kwargs = kwargs
        self.history = {'steps': []}
        for i in kwargs.keys():
            self.history.update({'val_'+i: []})
            self.history.update({'tra_'+i: []})
        self.valloader = None
        self.trainloader = None


    def save(self, step):
        short_history = {}
        for i in self.kwargs.keys():
            short_history.update({'val_'+i: []})
            short_history.update({'tra_'+i: []})
        k = 5
        short_train_set, waste = torch.utils.data.random_split(self.train_set, [k, len(self.train_set) - k])
        short_val_set, waste = torch.utils.data.random_split(self.val_set, [k, len(self.val_set) - k])
        self.valloader = DataLoader(dataset=short_val_set, batch_size=5, shuffle=True, num_workers=2)
        self.trainloader = DataLoader(dataset=short_train_set, batch_size=5, shuffle=True, num_workers=2)
        for i, ((val_in, val_label), (tra_in, tra_label)) in enumerate(zip(self.valloader, self.trainloader)):
            with torch.no_grad():
                self.model.eval()
                val_pred = self.model(val_in)
                tra_pred = self.model(tra_in)
                for j in self.kwargs.keys():
                    val_l = self.kwargs[j](val_pred, val_label).item()
                    tra_l = self.kwargs[j](tra_pred, tra_label).item()
                    short_history['val_'+j].append(val_l)
                    short_history['tra_'+j].append(tra_l)
                self.model.train()
        for i in self.kwargs.keys():
            self.history['val_' + i].append(sum(short_history['val_' + i]) / len(short_history['val_' + i]))
            self.history['tra_' + i].append(sum(short_history['tra_' + i]) / len(short_history['tra_' + i]))
        self.history['steps'].append(step)


    def plot(self):
        figures = []
        for i in self.kwargs.keys():
            fig, ax = plt.subplots()
            ax.plot(self.history['steps'], self.history['val_' + i], 'b')
            ax.plot(self.history['steps'], self.history['tra_' + i], 'r')
            print(f'{i}:')
            plt.show()
            figures.append(fig)
            plt.clf()
        return figures


def main(epochs=10, learning_rate=0.01, test_size=1000, train_batch_size=10, validation_batch_size=512, num_workers=2,
         loss=None, data_factor=1):
    if loss is None:
        loss = nn.MSELoss()  # TODO pass loss as function object
    url = 'https://raw.githubusercontent.com/Lokisfeuer/diamond/master/imdbdataset.csv'
    data = pd.read_csv('IMDB Dataset.csv')  # for jupyter: data = pd.read_csv(url)
    data = data.sample(frac=data_factor)
    data, sentiments = prepare_data(data)

    dataset = CustomMovieDataset(data, sentiments)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(data) - test_size, test_size])

    valloader = DataLoader(dataset=val_set, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers)
    dataloader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    model = NeuralNetwork(len(data[0]))
    r2loss = R2Score()
    mseloss = nn.MSELoss()
    bceloss = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    history = History(val_set, train_set, model, r2loss=r2loss, mseloss=mseloss, accuracy=get_acc, bceloss=bceloss)

    for epoch in range(epochs):
        print(f'Starting new batch {epoch + 1}/{epochs}')
        for step, (inputs, labels) in enumerate(dataloader):
            y_pred = model(inputs)
            for i in y_pred:
                if i > 1 or i < 0:
                    print('Warning Sigmoid ain\'t working.')
            l = loss(y_pred, labels)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (step + 1) % 5 == 0:  # if (step+1) % 100 == 0:
                history.save(epoch * len(dataloader) + step)
    history.plot()


def get_acc(pred, target):
    pred_tag = torch.round(pred)

    correct_results_sum = (pred_tag == target).sum().float()
    acc = correct_results_sum / target.shape[0]
    acc = torch.round(acc * 100)

    return acc


def bce_loss(pred, target):
    sum = 0.
    for p, t in zip(pred, target):
        sum += t * math.log(p) + (1 - t) * math.log(1 - p)
    sum = -1 * sum / len(target)
    return sum

def prepare_data(data):
    np_data = data.to_numpy().transpose()
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    reviews = model.encode(np_data[0])
    sentiments = np_data[1]
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
    kwargs = {
        'epochs':2,
        'learning_rate':0.01,
        'test_size':10, # 1000
        'train_batch_size':4,
        'validation_batch_size':512,
        'num_workers':2,
        'loss':nn.BCELoss(),
        'data_factor': 0.001
    }
    main(**kwargs)
    # for jupyter:
    #   change reading of csv
    #   adjust start command from jupyter.
