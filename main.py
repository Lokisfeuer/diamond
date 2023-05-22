import numpy
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import openai
from transformers import AutoTokenizer, AutoModel
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
import os
import pickle

openai.api_key = os.getenv('OPENAI_API_KEY')


class CustomTopicDataset(Dataset):
    def __init__(self, sentences, labels):
        self.x = sentences
        self.y = labels
        self.length = self.x.shape[0]
        self.shape = self.x[0].shape[0]
        self.feature_names = ['sentences', 'labels']

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
                    if len(val_pred) > 1:
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



def generate_data(data=None, topic='Biology', nr=15):
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
            requests = ask_ai(15*factor, f'Give me {15*factor} independent short requests about "{i}".\n\n1.')
            demands = ask_ai(15*factor, f'Give me {15*factor} independent short demands about "{i}".\n\n1.')
            questions = ask_ai(15*factor, f'Give me {15*factor} independent short questions about "{i}".\n\n1.')
            facts = ask_ai(5*factor, f'Give me {5*factor} independent short factual statements about "{i}".\n\n1.')
            sentences.extend(requests + demands + questions + facts)
        return sentences

    if data is None:
        all_sentences = []
        print(f'Writing sentences about {topic}.')
        # nr = 15
        fac = 1
        prompt = f'Give me {nr*2} independent keywords to the topic biology.\n\n1.'
        all_sentences.extend(gen_sentences(nr*2, fac, prompt))
        with open("save.p", "wb") as f:
            pickle.dump(all_sentences, f)
        print(f'Writing sentences not about {topic}.')
        prompt = f'Give me {nr} topics fully unrelated to biology.\n\n1.'
        all_sentences.extend(gen_sentences(nr, 2*fac, prompt))
        with open("save.p", "wb") as f:
            pickle.dump(all_sentences, f)
        print(all_sentences)
        print(len(all_sentences))
        print('Labelling sentences.')
        labels = []
        for i in range(len(all_sentences)):
            if i < len(all_sentences)/2:
                labels.append(True)
            else:
                labels.append(False)
        data = [all_sentences, labels]
        data = np.array(data).transpose()
        mapping = []
        uni = np.unique(data)
        for i in uni:
            mapping.append(np.where(data == i)[0][0])
        data = data[mapping[1:]]
        with open("save.p", "wb") as f:
            pickle.dump(data, f)
        print('full data has been saved to "save.p".')
    else:
        # data = data[~pd.isnull(data[:,0])]  # doesn't work
        mapping = []
        uni = np.unique(data)
        for i in uni:
            mapping.append(np.where(data == i)[0][0])
        data = data[mapping[1:]]
    pd.DataFrame(data).to_csv(f"{topic.replace(' ', '_')}_generated_data.csv", index = False, header = ['sentences', 'labels'])
    return pd.read_csv(f"{topic.replace(' ', '_')}_generated_data.csv")
    # TODO needs better filters and better quality. Especially empty inputs need to be filtered out!


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
    # test if this works with truncation=False

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def prepare_data_slowly(data):
    # data = data[data.review.str.split().str.len().le(64)]
    np_data = data.to_numpy().transpose()
    # use sentence embedding to encode the reviews
    # model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    embedded_data = np.array([[0,0]])
    # embedded_data = torch.load('embedded_data.pt')
    k = 100
    for i in range(round(len(np_data[0]) / k)):
        reviews = long_roberta(list(np_data[0][k*i:k*i+k]))
        labels = np_data[1][k*i:k*i+k]
        # reviews = torch.tensor_split(reviews, 0, dim=0)
        a = np.array([torch.tensor_split(reviews, len(reviews)), labels])
        a = a.transpose()
        embedded_data = np.append(embedded_data, a, axis=0)
        if i == 0:
            embedded_data = embedded_data[1:]
        # embedded_data.extend(a)
        torch.save(embedded_data, 'embedded_data.pt')
        print(f'saved {i+1} / {len(np_data[0]) / k}')
    return embedded_data.transpose()


def check_length(data):
    def tokenize(sentences):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
        encoded_input = tokenizer(sentences, padding=True, truncation=False, return_tensors='pt')
        return encoded_input

    sorted_data = data.reindex(data.sentences.str.len().sort_values().index[::-1]).reset_index(drop=True)

    for idx, row in sorted_data.iterrows():
        length = len(tokenize(row.sentences)['input_ids'][0])
        if length > 512:
            print('Warning: Paragraph longer than 512 tokens therefore to long.')
        elif length > 128:
            print('Warning: Paragraph longer than 128 tokens therefore longer than recommended.')
        elif length < 80:
            break



def analyse_full_data(data):
    '''
    max = data.review.str.len().sum()
    print('Full set average length')
    print(max / 50000)
    data = data[data.review.str.split().str.len().le(64)]
    max = data.review.str.len().sum()
    print('Short set average length')
    print(max / 50000)
    '''

    print('INFO')
    data.info()
    data.groupby(['labels']).describe()
    print(f'Number of unique sentences: {data["sentences"].nunique()}')
    duplicates = data[data.duplicated()]
    print(f'Number of duplicate rows:\n{len(duplicates)}')
    print(f'Check for nulls: {data.isnull().sum()}')
    sns.countplot(x=data['labels'])  # ploting distribution for easier understanding
    print(data.head(3))

    # let's see how data is looklike
    random_index = random.randint(0, data.shape[0] - 3)
    for row in data[['sentences', 'labels']][random_index:random_index + 3].itertuples():
        _, text, label = row
        class_name = "About topic"
        if label == 0:
            class_name = "Not about topic"
        display(HTML(f"<h5><b style='color:red'>Text: </b>{text}</h5>"))
        display(HTML(f"<h5><b style='color:red'>Target: </b>{class_name}<br><hr></h5>"))
    # data contain so much garbage needs to be cleaned

    truedata = data[data['labels'] == 1]
    truedata = truedata['sentences']
    falsedata = data[data['labels'] == 0]
    falsedata = falsedata['sentences']

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
    wordcloud_draw(truedata, 'white', 'Most-common words about the topic')

    plt.subplot(1, 2, 2)
    wordcloud_draw(falsedata, 'white', 'Most-common words not about the topic')
    plt.show() # end wordcloud

    data['text_word_count'] = data['sentences'].apply(lambda x: len(x.split()))

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
        sns.histplot(data=data, x=col, hue='labels', bins=50)
        plt.title(f"Distribution of Various word counts with respect to target")
    plt.tight_layout()
    plt.show()


class TopicIdentifier:
    def __init__(self):
        self.running_loss = None
        self.optimizer = None
        self.dataloader = None
        self.model = None
        self.loss = None
        self.dataframe = None
        self.val_set = None
        self.train_set = None
        self.labels = None
        self.sentences = None
        self.embedded_data = None
        self.raw_data = None
        self.dataset = None

    def generate_training_data(self, topic):
        data = load_data('save.p')
        self.raw_data = generate_data(data=data, topic=topic, nr=15)

    def embedd_data(self):
        def get_element(arr):
            return arr[0]
        # self.embedded_data = prepare_data_slowly(self.raw_data)
        self.embedded_data = torch.load('embedded_data.pt').transpose()
        tpl = tuple(map(get_element, tuple(numpy.array_split(self.embedded_data[0], len(self.embedded_data[0])))))
        self.sentences = torch.cat(tpl)
        self.labels = self.embedded_data[1]
        self.labels[self.labels == True] = 1.
        self.labels[self.labels == False] = 0.
        self.labels = numpy.expand_dims(self.labels, axis=1).astype('float32')
        self.labels = torch.from_numpy(self.labels)
        self.dataset = CustomTopicDataset(self.sentences, self.labels)

    def analyse_training_data(self):
        check_length(self.raw_data)
        analyse_full_data(self.raw_data)

    def train(self, epochs, lr=0.01, val_frac=0.1, batch_size=25, loss=nn.BCELoss()):
        def get_acc(pred, target):
            pred_tag = torch.round(pred)

            correct_results_sum = (pred_tag == target).sum().float()
            acc = correct_results_sum / target.shape[0]
            acc = torch.round(acc * 100)

            return acc

        val_len = int(round(len(self.dataset)*val_frac))
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [len(self.dataset)-val_len, val_len])
        self.dataloader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=True)
        self.model = NeuralNetwork(self.dataset.shape)
        
        self.loss = loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        r2loss = R2Score()
        mseloss = nn.MSELoss()
        bceloss = nn.BCELoss()
        accuracy = get_acc

        history = History(self.val_set, self.train_set, self.model, r2loss=r2loss, mseloss=mseloss, accuracy=accuracy, bceloss=bceloss)

        # main training loop
        for epoch in range(epochs):
            self.running_loss = 0.
            print(f'Starting new batch {epoch + 1}/{epochs}')
            for step, (inputs, labels) in enumerate(self.dataloader):
                y_pred = self.model(inputs)
                l = self.loss(y_pred, labels)
                l.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.running_loss += l.item()
                if (step + 1) % 100 == 0:  # if (step+1) % 100 == 0:
                    print(f'current loss:\t\t{self.running_loss / 100}')
                    self.running_loss = 0
                    history.save(epoch * len(self.dataloader) + step)
                    # save current state of the model to history
        return history

def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    ti = TopicIdentifier()
    ti.generate_training_data('biology')
    # ti.analyse_training_data()
    ti.embedd_data()
    history = ti.train(epochs=5, lr=0.001, val_frac=0.1, batch_size=5, loss=nn.BCELoss())
    history.plot()
