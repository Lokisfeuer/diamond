# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import numpy as np
import os
import math
import random
from datetime import datetime as d
import jupyter
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as f
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
# api.authenticate() # comment out for jupyter
import pandas as pd
from torchvision.io import read_image
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import R2Score
# from torchmetrics.functional import r2_score

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


def main(epochs=10, learning_rate=0.01, test_size=1000, train_batch_size=10, validation_batch_size=512, num_workers=2, loss=None, optimizer=None, data_factor=1):
    if loss is None:
        loss = 'nn.MSELoss()'
    if optimizer is None:
        optimizer = 'torch.optim.SGD(model.parameters(), lr=learning_rate)'
        
    writer = SummaryWriter('runs/diamond')
    url = 'https://raw.githubusercontent.com/Lokisfeuer/diamond/master/diamonds.csv'
    data = pd.read_csv('diamonds.csv')  # for jupyter: data = pd.read_csv(url)
    data = data.sample(frac=data_factor)
    data, prices, maxi, mini = normalize_data(data)
    # data = data[:100]
    # prices = prices[:100]
    test_input = torch.tensor(np.array([data[500]]), dtype=torch.float32)
    price = get_real_price(prices[500][0], maxi, mini)
    data = torch.tensor(data, dtype=torch.float32)
    prices = torch.tensor(prices, dtype=torch.float32)
    dataset = CustomDiamondDataset(data, prices)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(data)-test_size, test_size])

    valloader = DataLoader(dataset=val_set, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers)
    dataloader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    model = NeuralNetwork(len(data[0]))
    loss = eval(loss)
    r2loss = R2Score()
    mseloss = nn.MSELoss()
    optimizer = eval(optimizer)
    # loss = nn.MSELoss()  # try others: r squared metric scale from -1 (opposite) to 1 (ideal) to infinite (wrong again); accuracy error
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(f'Prediction beforehand: {get_real_price(model(test_input).item(), maxi, mini)}\t\tcorrect was: {price}')
    running_loss = 0.
    running_r2loss = 0.
    val_mse_loss = []
    val_r2loss = []
    val_real_price_percentage_loss = []
    percentages = []
    training_mse_loss = []
    training_r2loss = []
    training_real_price_percentage_loss = []
    x_axis = []
    for epoch in range(epochs):
        print(f'Starting new batch {epoch+1}/{epochs}')
        # check_accuracy(valloader, model, maxi, mini)
        for step, (inputs, labels) in enumerate(dataloader):
            # calculate r squarred loss
            y_pred = model(inputs)
            for pred, label in zip(y_pred, labels):
                pr = get_real_price(pred, maxi, mini)
                la = get_real_price(label, maxi, mini)
                percentages.append(abs(pr-la)/la)
            l = loss(y_pred, labels)
            # msel = mseloss(y_pred, labels)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += l.item()  # l.item()
            with torch.no_grad():
                model.eval()
                x = r2loss(y_pred, labels).item()
                if x >= 1:
                    print('x >= 1')
                running_r2loss += x
                model.train()
            if (step+1) % 100 == 0: # if (step+1) % 100 == 0:
                mse_l, r2_l, percent_l = evaluate_model(model, valloader, maxi, mini, loss, r2loss)
                val_mse_loss.append(mse_l)
                val_r2loss.append(r2_l)
                val_real_price_percentage_loss.append(percent_l)
                training_mse_loss.append(running_loss / 100)
                training_r2loss.append(running_r2loss / 100)
                training_real_price_percentage_loss.append(sum(percentages)/len(percentages)*100) # to static. Why?
                x_axis.append(epoch*len(dataloader) + step)
                writer.add_scalar('training_mse_loss', running_loss / 100, epoch*len(dataloader) + step)
                writer.add_scalar('training_real_price_percentage_loss', sum(percentages)/len(percentages)*100, epoch*len(dataloader) + step)
                running_loss = 0.
                running_r2loss = 0.
                percentages = []
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
        }
        torch.save(checkpoint, 'checkpoint.pth')
        # to load:
        # loaded_checkpoint =torch.load('checkpoint.pth')
    writer.close()
    args = [x_axis, val_mse_loss, training_mse_loss, val_r2loss, training_r2loss, val_real_price_percentage_loss, training_real_price_percentage_loss]
    print(args)
    graph(*args)
    print(f'Prediction afterwards: {get_real_price(model(test_input).item(), maxi, mini)}\t\tcorrect was: {price}')
    file = '2model.pth'
    torch.save(model.state_dict(), file)

def graph(x_axis, val_mse_loss, training_mse_loss, val_r2loss, training_r2loss, val_real_price_percentage_loss, training_real_price_percentage_loss):
    path = os.path.abspath(os.getcwd())
    ra = str(random.randint(1,100))
    ver = str(2)
    now = str(d.now().isoformat()).replace(':', 'I').replace('.', 'i')
    fig, ax = plt.subplots()
    ax.plot(x_axis, val_mse_loss, 'b') # ? (0)
    ax.plot(x_axis, training_mse_loss, 'r')  # 0
    plt.savefig(f'plots/{ver}mse_ver{ra}_{now}.png')
    fig, ax = plt.subplots()
    ax.plot(x_axis, val_r2loss, 'b') # ? (0)
    ax.plot(x_axis, training_r2loss, 'r') # ? (0)
    plt.savefig(f'plots/{ver}r2_ver{ra}_{now}.png')
    fig, ax = plt.subplots()
    ax.plot(x_axis, val_real_price_percentage_loss, 'b') # good
    ax.plot(x_axis, training_real_price_percentage_loss, 'r') # good
    plt.savefig(f'plots/{ver}perc_ver{ra}_{now}.png')

def load_model(data, file = '1model.pth'):
    valloader = DataLoader(dataset=val_set, batch_size=512, shuffle=True, num_workers=2)

    loaded_model = NeuralNetwork(len(data[0]))
    loaded_model.load_state_dict(torch.load(file))
    return loaded_model


def evaluate_model(model, valloader, maxi, mini, loss, r2loss):
    # print('\n\nStart evaluating')
    with torch.no_grad():
        # try using accuracy in addition to loss
        model.eval()
        percentages = []
        avg_mse_loss = []
        avg_r2_loss = []
        for step, (inputs, labels) in enumerate(valloader):
            mistakes = []
            y_pred = model(inputs)
            for pred, label in zip(y_pred, labels):
                pr = get_real_price(pred, maxi, mini)
                la = get_real_price(label, maxi, mini)
                # print(f'Estimation: {p}; True: {la}')
                mistakes.append(abs(pr-la))
                percentages.append(abs(pr-la)/la)
            l = loss(y_pred, labels)
            r2l = r2loss(y_pred, labels)
            # print(f'Average real-price error for this batch was: \t\t\t\t\t{sum(mistakes)/len(mistakes)}.')
            # print(f'Average real-price error relative to the price in percent was: '
            #       f'\t{sum(percentages)/len(percentages)*100}%.')
            # print(f'Average loss for this batch was \t\t\t\t\t\t\t\t{l.item()}\n')
            avg_mse_loss.append(l.item())
            avg_r2_loss.append(r2l.item())
        model.train()
        return sum(avg_mse_loss)/len(avg_mse_loss), sum(avg_r2_loss)/len(avg_r2_loss), sum(percentages)/len(percentages)*100

    # Graph test over training !
    # plot everything on the graph, accuracy, MSEloss, R^2loss, percentage_price%



# check accuracy causes Error - not used
def check_accuracy(loader, model, maxi, mini):
    model.eval()
    with torch.no_grad():
        aver = []
        for x, y in loader:
            correct = get_real_price(y.item(), maxi, mini)
            resp = model(x)
            price = get_real_price(resp.item(), maxi, mini)
            aver.append(abs(correct - price))
        model.train()
        print(sum(aver)/len(aver))
        return sum(aver)/len(aver)
            #scores = model(x)
            #res = scores.unsqueeze(1) - y
            #a = torch.mean(res).item()
            #aver.append(a)

            #_, predictions = scores.max(1)
            #num_correct += (predictions == y).sum()
            #num_samples += predictions.size(0)

        # print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')


# both robertas fully copied from https://huggingface.co/sentence-transformers/all-roberta-large-v1
def short_roberta(sentences):
    # sentences = ["This is an example sentence", "Each sentence is converted"]

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
    sentence_embeddings = f.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def get_real_price(val, maxi, mini):
    x = (maxi-mini) * val + mini
    return math.exp(x)


def normalize_data(data):
    # max / min normalization the dataset
    # scalars stay normal columns, categories get different columns - one hot encoding
    # price on a logarythmic scale

    '''
    let's build a tensor with the following dimensions:
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
    kwargs = {
        'epochs':10,
        'learning_rate':0.01,
        'test_size':1000, # 1000
        'train_batch_size':10,
        'validation_batch_size':512,
        'num_workers':2,
        'loss':'nn.MSELoss()',
        'optimizer':'torch.optim.SGD(model.parameters(), lr=learning_rate)',
        'data_factor': 1
    }
    main(**kwargs)
    #args = [[29, 59, 89], [0.04725663047283888, 0.04288289994001389, 0.03785799648612738], [0.03140254817903042, 0.01449822638183832, 0.013357452619820832], [0.21161172389984131, 0.3071813404560089, 0.3442371547222137], [-0.1830847430229187, 0.05015255331993103, 0.08432324945926667], [86.34664962859036, 78.04877220648744, 74.22272207896643], [67.15759899285841, 91.4195255019661, 84.78499436740307]]
    #graph(*args)

