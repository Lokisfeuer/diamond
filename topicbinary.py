'''
# These are the start commands necessary when running this as jupyter notebook on colabs:

print('starting.')
!pip install transformers
!pip install sentence_transformers
!pip install torchmetrics
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
'''
import math

# import everything necessary
import numpy as np  # for data preparation
from sentence_transformers import SentenceTransformer  # for sentence embedding in data preparation
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # for plotting the history graphs
import pandas as pd  # to read data in from csv
from torchmetrics import R2Score
import openai
import os
import pickle

openai.api_key = os.getenv('OPENAI_API_KEY')


# The Dataset class
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


# This is the history class to monitor different metrics during training
class History:
    def __init__(self, val_set, train_set, model, **kwargs):  # kwargs are the metrics being monitored
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

    # save current value of metrics of model.
    def save(self, step):
        short_history = {}
        for i in self.kwargs.keys():
            short_history.update({'val_'+i: []})
            short_history.update({'tra_'+i: []})
        k = 5
        # generate random datasubsets and loaders with k entries for training and validation
        short_train_set, waste = torch.utils.data.random_split(self.train_set, [k, len(self.train_set) - k])
        short_val_set, waste = torch.utils.data.random_split(self.val_set, [k, len(self.val_set) - k])
        self.valloader = DataLoader(dataset=short_val_set, batch_size=5, shuffle=True, num_workers=2)
        self.trainloader = DataLoader(dataset=short_train_set, batch_size=5, shuffle=True, num_workers=2)
        # iterate over the dataloaders
        for i, ((val_in, val_label), (tra_in, tra_label)) in enumerate(zip(self.valloader, self.trainloader)):
            with torch.no_grad():
                self.model.eval()
                val_pred = self.model(val_in)
                tra_pred = self.model(tra_in)
                for j in self.kwargs.keys():
                    # calculate each metric of kwargs and append value to their short history
                    val_l = self.kwargs[j](val_pred, val_label).item()
                    tra_l = self.kwargs[j](tra_pred, tra_label).item()
                    short_history['val_'+j].append(val_l)
                    short_history['tra_'+j].append(tra_l)
                self.model.train()
        for i in self.kwargs.keys():
            # for each metric average over their short history and append to the history
            self.history['val_' + i].append(sum(short_history['val_' + i]) / len(short_history['val_' + i]))
            self.history['tra_' + i].append(sum(short_history['tra_' + i]) / len(short_history['tra_' + i]))
        self.history['steps'].append(step)


    # plot the history in one graph per metric using red for training data and blue for validation data.
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
        loss = nn.MSELoss()

    # load and prepare data
    url = 'https://raw.githubusercontent.com/Lokisfeuer/diamond/master/imdbdataset.csv'
    data = pd.read_csv('IMDB Dataset.csv')  # for jupyter: data = pd.read_csv(url)
    data = data.sample(frac=data_factor)
    data, sentiments = prepare_data(data)

    dataset = CustomMovieDataset(data, sentiments)
    # split data into train_set and val_set
    train_set, val_set = torch.utils.data.random_split(dataset, [len(data) - test_size, test_size])

    # define dataloaders, model and metrics
    valloader = DataLoader(dataset=val_set, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers)
    dataloader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    model = NeuralNetwork(len(data[0]))
    r2loss = R2Score()
    mseloss = nn.MSELoss()
    bceloss = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # define history object with metrics to monitor
    history = History(val_set, train_set, model, r2loss=r2loss, mseloss=mseloss, accuracy=get_acc, bceloss=bceloss)

    # main training loop
    for epoch in range(epochs):
        print(f'Starting new batch {epoch + 1}/{epochs}')
        for step, (inputs, labels) in enumerate(dataloader):
            y_pred = model(inputs)
            l = loss(y_pred, labels)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (step + 1) % 100 == 0:  # if (step+1) % 100 == 0:
                history.save(epoch * len(dataloader) + step)
                # save current state of the model to history
    history.plot()


# this is one of the custom metrics being monitored. Accuracy in percentage of correctly classified.
def get_acc(pred, target):
    pred_tag = torch.round(pred)

    correct_results_sum = (pred_tag == target).sum().float()
    acc = correct_results_sum / target.shape[0]
    acc = torch.round(acc * 100)

    return acc

def prepare_data(data):
    np_data = data.to_numpy().transpose()
    # use sentence embedding to encode the reviews
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    reviews = model.encode(np_data[0])
    # encode the sentiments
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
    return reviews, sentiments  # return torch tensors.

def generate_data(data=None, topic='Biology'):
    def ask_ai(nr, prompt):
        response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=1, max_tokens=10*nr)
        response = '1.' + response['choices'][0]['text'] + '\n'
        l = []
        for i in range(nr):
            pos = response.find(str(i + 1))
            beg = pos + len(str(i + 1)) + 2
            end = response[beg:].find('\n')
            l.append(response[beg:beg + end])
        return l

    def gen_sentences(nr, factor, prompt):
        keywords = ask_ai(nr, prompt)
        sentences = []
        for i in keywords:
            print(i)
            requests = ask_ai(15*factor, f'Give me {15*factor} independent requests about "{i}".\n\n1.')
            demands = ask_ai(15*factor, f'Give me {15*factor} independent demands about "{i}".\n\n1.')
            questions = ask_ai(15*factor, f'Give me {15*factor} independent questions about "{i}".\n\n1.')
            facts = ask_ai(5*factor, f'Give me {5*factor} independent factual statements about "{i}".\n\n1.')
            sentences.extend(requests + demands + questions + facts)
        return sentences

    if data is None:
        all_sentences = []
        print(f'Writing sentences about {topic}.')
        prompt = 'Give me 10 independent keywords to the topic biology.\n\n1.'
        all_sentences.extend(gen_sentences(10, 1, prompt))
        with open("save.p", "wb") as f:
            pickle.dump(all_sentences, f)
        print(f'Writing sentences not about {topic}.')
        prompt = 'Give me 5 topics fully unrelated to biology.\n\n1.'
        all_sentences.extend(gen_sentences(5, 2, prompt))
        with open("save.p", "wb") as f:
            pickle.dump(all_sentences, f)
        print(all_sentences)
        print(len(all_sentences))
        print('Labelling sentences.')
        labels = []
        for i in range(len(all_sentences)):
            if i < 400:  # if i < len(all_sentences)/2:
                labels.append(True)
            else:
                labels.append(False)
        data = [all_sentences, labels]
        data = np.array(data).transpose()
        with open("save.p", "wb") as f:
            pickle.dump(data, f)
        print('full data has been saved to "save.p".')
    pd.DataFrame(data).to_csv("test.csv", index = False, header = ['sentence', f'about {topic}'])
    return pd.read_csv("test.csv")



def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(data)
    print(f'Len data: {len(data)}')
    return data


if __name__ == '__main__':
    # call main function with these parameters:
    kwargs = {
        'epochs':10,
        'learning_rate':0.01,
        'test_size':1000, # 1000
        'train_batch_size':25,
        'validation_batch_size':512,
        'num_workers':2,
        'loss':nn.BCELoss(),
        'data_factor': 1  # between 0 and 1, how much of the data is used.
    }
    # main(**kwargs)
    # To run this as jupyter notebook change the following:
    #   change reading of csv
    #   uncomment the blockstring at the beginning of the file
    data = load_data('save.p')
    print(generate_data(data=data))
