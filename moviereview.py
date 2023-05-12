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
# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi()
# api.authenticate() # comment out for jupyter
import pandas as pd
from torchvision.io import read_image
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import R2Score
# from torchmetrics.functional import r2_score


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
            self.history.update({'val_'+i: []})
            self.history.update({'tra_'+i: []})
        k = 5
        perm = torch.randperm(self.train_set.size(0))
        short_train_set = self.train_set[perm[:k]]
        perm = torch.randperm(self.val_set.size(0))
        short_val_set = self.val_set[perm[:k]]
        print(short_train_set)
        print(short_val_set)
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


    writer = SummaryWriter('runs/diamond')
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
    bceloss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    history = History(val_set, train_set, model, r2loss=r2loss, mseloss=mseloss, accuracy=get_acc, bceloss=bceloss)

    val_mse_loss = []
    val_r2loss = []
    val_real_price_percentage_loss = []
    percentages = []
    training_mse_loss = []
    training_r2loss = []
    training_real_price_percentage_loss = []
    x_axis = []
    for epoch in range(epochs):
        running_loss = 0.
        running_r2loss = 0.
        print(f'Starting new batch {epoch + 1}/{epochs}')
        # check_accuracy(valloader, model, maxi, mini)
        for step, (inputs, labels) in enumerate(dataloader):
            # calculate r squarred loss
            y_pred = model(inputs)
            l = loss(y_pred, labels)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += l.item()  # l.item()
            with torch.no_grad():
                model.eval()
                msel = mseloss(y_pred, labels)
                x = r2loss(y_pred, labels).item()
                running_r2loss += x
                model.train()
            if (step + 1) % 100 == 0:  # if (step+1) % 100 == 0:
                history.save(epoch * len(dataloader) + step)
                mse_l, r2_l, percent_l = evaluate_model(model, valloader, loss, r2loss)
                val_mse_loss.append(mse_l)
                val_r2loss.append(r2_l)
                val_real_price_percentage_loss.append(percent_l)
                training_mse_loss.append(running_loss / 100)
                training_r2loss.append(running_r2loss / 100)
                training_real_price_percentage_loss.append(sum(percentages) / len(percentages) * 100)  # to static. Why?
                x_axis.append(epoch * len(dataloader) + step)
                writer.add_scalar('training_mse_loss', running_loss / 100, epoch * len(dataloader) + step)
                writer.add_scalar('training_real_price_percentage_loss', sum(percentages) / len(percentages) * 100,
                                  epoch * len(dataloader) + step)
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
    args = [x_axis, val_mse_loss, training_mse_loss, val_r2loss, training_r2loss, val_real_price_percentage_loss,
            training_real_price_percentage_loss]
    print(args)
    # graph(*args)
    history.plot()
    file = '2model.pth'
    torch.save(model.state_dict(), file)

def evaluate_model(model, valloader, loss, r2loss):
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


def get_acc(pred, target):
    pred_tag = torch.round(pred)

    correct_results_sum = (pred_tag == target).sum().float()
    acc = correct_results_sum / target.shape[0]
    acc = torch.round(acc * 100)

    return acc


def graph(x_axis, val_mse_loss, training_mse_loss, val_r2loss, training_r2loss, val_real_price_percentage_loss,
          training_real_price_percentage_loss):
    path = os.path.abspath(os.getcwd())
    ra = str(random.randint(1, 100))
    ver = str(2)
    now = str(d.now().isoformat()).replace(':', 'I').replace('.', 'i')
    fig, ax = plt.subplots()
    ax.plot(x_axis, val_mse_loss, 'b')  # ? (0)
    ax.plot(x_axis, training_mse_loss, 'r')  # 0
    print('mse loss:')
    plt.show()
    plt.savefig(f'plots/{ver}mse_ver{ra}_{now}.png')  # comment out for jupyter
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(x_axis, val_r2loss, 'b')  # ? (0)
    ax.plot(x_axis, training_r2loss, 'r')  # ? (0)
    print('r2 loss:')
    plt.show()
    plt.savefig(f'plots/{ver}r2_ver{ra}_{now}.png')  # comment out for jupyter
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(x_axis, val_real_price_percentage_loss, 'b')  # good
    ax.plot(x_axis, training_real_price_percentage_loss, 'r')  # good
    print('real price percentage loss:')
    plt.show()
    plt.savefig(f'plots/{ver}perc_ver{ra}_{now}.png')  # comment out for jupyter
    plt.clf()


def short_roberta(sentences):
    # sentences = ["This is an example sentence", "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."]
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    embeddings = model.encode(sentences)
    return embeddings


# not used
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


def prepare_data(data):
    np_data = data.to_numpy().transpose()
    reviews = short_roberta(np_data[0])
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

    # TODO better save scaler as an object
    scaler = MinMaxScaler()
    data = pd.DataFrame(new_array)
    prices = pd.DataFrame(prices)
    normalized_data = scaler.fit_transform(data)
    normalized_prices = scaler.fit_transform(prices)

    return normalized_data, normalized_prices, maxi, mini


if __name__ == '__main__':

    kwargs = {
        'epochs':1,
        'learning_rate':0.01,
        'test_size':10, # 1000
        'train_batch_size':10,
        'validation_batch_size':512,
        'num_workers':2,
        'loss':nn.CrossEntropyLoss(),
        'data_factor': 0.001
    }
    # short_roberta('')
    main(**kwargs)
    #args = [[29, 59, 89], [0.04725663047283888, 0.04288289994001389, 0.03785799648612738], [0.03140254817903042, 0.01449822638183832, 0.013357452619820832], [0.21161172389984131, 0.3071813404560089, 0.3442371547222137], [-0.1830847430229187, 0.05015255331993103, 0.08432324945926667], [86.34664962859036, 78.04877220648744, 74.22272207896643], [67.15759899285841, 91.4195255019661, 84.78499436740307]]
    #graph(*args)

    # for jupyter:
    #   comment out saving of graph (and not model?).
    #   change reading of csv
    #   adjust start command from jupyter.
