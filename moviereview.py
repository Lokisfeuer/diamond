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

import sys
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import random
import seaborn as sns
from IPython.display import HTML, display
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords


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
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
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
        k = 500
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
    data = pd.read_csv(url)
    # data = data.sample(frac=data_factor)
    data, sentiments = prepare_data(data)

    dataset = CustomMovieDataset(data, sentiments)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(data) - test_size, test_size])
    print(len(val_set))
    print(len(train_set))

    valloader = DataLoader(dataset=val_set, batch_size=validation_batch_size, shuffle=True)
    dataloader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True)
    print(data)
    model = NeuralNetwork(len(data[0]))
    r2loss = R2Score()
    mseloss = nn.MSELoss()
    bceloss = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = History(val_set, train_set, model, r2loss=r2loss, mseloss=mseloss, accuracy=get_acc, bceloss=bceloss)

    for epoch in range(epochs):
        running_l = 0
        print(f'Starting new batch {epoch + 1}/{epochs}')
        for step, (inputs, labels) in enumerate(dataloader):
            y_pred = model(inputs)
            # l = loss(y_pred, labels)
            l = mseloss(y_pred, labels)
            running_l += l.item()
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (step + 1) % 50 == 0:  # if (step+1) % 100 == 0:
                history.save(epoch * len(dataloader) + step)
                print(f'training loss: {running_l / 50}')
                running_l = 0
    history.plot()


def get_acc(pred, target):
    pred_tag = torch.round(pred)

    correct_results_sum = (pred_tag == target).sum().float()
    acc = correct_results_sum / target.shape[0]
    acc = torch.round(acc * 100)

    return acc


def prepare_data(data):
    np_data = data.to_numpy().transpose()
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
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
    rev = np.array(file_data, dtype=np.float32)
    analyse_data(rev, sentiments)
    sys.exit()
    sentiments = torch.from_numpy(sentiments)
    reviews = torch.tensor(reviews, dtype=torch.float32)
    sentiments = torch.tensor(sentiments, dtype=torch.float32)  # line needed? dtype?
    return reviews, sentiments

def prepare_data_slowly(data=None):
    if data is None:
        url = 'https://raw.githubusercontent.com/Lokisfeuer/diamond/master/imdbdataset.csv'
        data = pd.read_csv(url)
    np_data = data.to_numpy().transpose()
    # use sentence embedding to encode the reviews
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    all_reviews = []
    k = 100
    for i in range(round(len(np_data[0]) / k)):
        reviews = model.encode(np_data[0][k*i:k*i+k])
        all_reviews.append(reviews)
        torch.save(all_reviews, 'embedded_reviews.pt')
        print(f'saved {i+1} / {len(np_data[0]) / k}')

def long_roberta():
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
    model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=False, return_tensors='pt')
    # TODO: clarification on the truncation. Does it throw an error (later)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    print("Sentence embeddings:")
    print(sentence_embeddings)

def analyse_full_data(dataset):
    url = 'https://raw.githubusercontent.com/Lokisfeuer/diamond/master/imdbdataset.csv'
    data = pd.read_csv(url)
    max = data.review.str.len().sum()
    print('Full set average length')
    print(max / 50000)
    data = data[data.review.str.split().str.len().le(64)]
    max = data.review.str.len().sum()
    print('Short set average length')
    print(max / 50000)



    print('INFO')
    data.info()
    data.groupby(['sentiment']).describe()
    print(f'Number of unique reviews: {data["review"].nunique()}')
    duplicates = data[data.duplicated()]
    print(f'Number of duplicate rows:\n{len(duplicates)}')
    print(f'Check for nulls: {data.isnull().sum()}')
    sns.countplot(x=data['sentiment'])  # ploting distribution for easier understanding
    print(data.head(3))

    # let's see how data is looklike
    random_index = random.randint(0, data.shape[0] - 3)
    for row in data[['review', 'sentiment']][random_index:random_index + 3].itertuples():
        _, text, label = row
        class_name = "Positive"
        if label == 0:
            class_name = "Negative"
        display(HTML(f"<h5><b style='color:red'>Text: </b>{text}</h5>"))
        display(HTML(f"<h5><b style='color:red'>Target: </b>{class_name}<br><hr></h5>"))
    # data contain so much garbage needs to be cleaned

    positivedata = data[data['sentiment'] == 'positive']
    positivedata = positivedata['review']
    negdata = data[data['sentiment'] == 'negative']
    negdata = negdata['review']

    def wordcloud_draw(data, color, s):
        words = ' '.join(data)
        cleaned_word = " ".join([word for word in words.split() if (word != 'movie' and word != 'film')])
        wordcloud = WordCloud(stopwords=stopwords.words('english'), background_color=color, width=2500,
                              height=2000).generate(cleaned_word)
        plt.imshow(wordcloud)
        plt.title(s)
        plt.axis('off')

    plt.figure(figsize=[20, 10])
    plt.subplot(1, 2, 1)
    wordcloud_draw(positivedata, 'white', 'Most-common Positive words')

    plt.subplot(1, 2, 2)
    wordcloud_draw(negdata, 'white', 'Most-common Negative words')
    plt.show() # end wordcloud

    data['text_word_count'] = data['review'].apply(lambda x: len(x.split()))

    numerical_feature_cols = ['text_word_count']

    plt.figure(figsize=(20, 3))
    for i, col in enumerate(numerical_feature_cols):
        plt.subplot(1, 3, i + 1)
        sns.histplot(data=data, x=col, bins=50, color='#6495ED')
        plt.title(f"Distribution of Various word counts")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 3))
    for i, col in enumerate(numerical_feature_cols):
        plt.subplot(1, 3, i + 1)
        sns.histplot(data=data, x=col, hue='sentiment', bins=50)
        plt.title(f"Distribution of Various word counts with respect to target")
    plt.tight_layout()
    plt.show()




analyse_full_data()
if __name__ == '__main__':
    # prepare_data_slowly()
    kwargs = {
        'epochs':10,
        'learning_rate':0.01,
        'test_size':500, # 1000  # 10% of full dataset
        'train_batch_size':25,
        'validation_batch_size':512,
        'num_workers':2,
        'loss':nn.BCELoss(),
        'data_factor': 1
    }
    # main(**kwargs)
    # for jupyter:
    #   change reading of csv
    #   adjust start command from jupyter.


'''
Introduction

Prompt engineering
Analyse dataset
results of my network in beautiful graphs

Speed comparison between my network and prompt engineering

CONCRETE:
Hello Evererybody, here I show you how I've spend my time this week, what I've learned and what I've achieved

Started with: (All purely training)
    Diamond
        Goal: Estimate diamond price by multiple properties-
        Challenge: prepare data with:
            one-hot encoding
            min max
            logarythmic scale for prices
        result: Graph
    Movie review sentiment - fail
        Goal: classfiy a movie review as positive or negative
        Challenge: Embedd text data, ?
        Result: Graphs from chat with Zak, no success.
            maybe inputs to long?
    Twitter review sentiment
        Goal: classify sentiment as positive or negative
        Challenge: Analyse Dataset fully and find problem with movie review.
            Shorter inputs, balanced, very large dataset (1.6 million)
        Result: ?
Topic identification
    class and its functions
    data generation loop
        experience
            examples
            pretty repetitive -> can easily be sorted
            -> use of keywords and subtopics
            more diversity in false prompts needed
        solution?
        numbers of how much and average percentage of duplicates
    analyse data function
        check length
            why? because of possible truncation -> as learned from movie reviews
        rest of analyse
            link where I got it from
            examples of analysation
    data is embedded
    model is trained
        history object keeps track of everything and is returned as well as the full trained model
        model is automatically saved.
    outcome:
        examples of graphs as well as live show of models to different topics. 
        comparison to prompt engineering
    
Therefore todo:
    Reproduce movie review graphs
    Analyse, ideally fully solve movie reviews
    improve and then update the generate data function
    analyse generated datasets to biology and chemistry and write them to data generation.
    compare my program to prompt engineering and analyse accuracy
    prepare live presentation for biology and chemistry
    Graph training for chemistry and biology.
    
    Comment through the code!
    

'''