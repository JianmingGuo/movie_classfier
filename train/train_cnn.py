import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from string import punctuation
from gensim.models import KeyedVectors

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

style.use('seaborn-poster')
style.use('ggplot')


with open('content/movie_prediction/train_np_imgs_norm','rb') as f: X_img_train = pickle.load(f)
with open('content/movie_prediction/test_np_imgs_norm', 'rb') as f: X_img_test = pickle.load(f)
with open('content/movie_prediction/val_np_imgs_norm', 'rb') as f: X_img_val = pickle.load(f)

dataset = pd.read_csv("content/movie_prediction/dataset_mod.csv")
train = pd.read_csv("content/movie_prediction/train_data.csv")
test = pd.read_csv("content/movie_prediction/test_data.csv")
val = pd.read_csv("content/movie_prediction/val_data.csv")

dataset['genre_list'] = dataset['genre_list'].apply(lambda x: ast.literal_eval(x))
train['genre_list'] = train['genre_list'].apply(lambda x: ast.literal_eval(x))
test['genre_list'] = test['genre_list'].apply(lambda x: ast.literal_eval(x))
val['genre_list'] = val['genre_list'].apply(lambda x: ast.literal_eval(x))

labels = {}

for genre in test['genre_list']:
    if len(genre) in labels:
        labels[len(genre)] += 1
    else:
        labels[len(genre)] = 1


mlb = MultiLabelBinarizer()
mlb.fit(dataset['genre_list'].tolist())

transformed_labels = mlb.fit_transform(dataset['genre_list'].tolist())

train_labels = mlb.transform(train['genre_list'].tolist())

test_labels = mlb.transform(test['genre_list'].tolist())

val_labels = mlb.transform(val['genre_list'].tolist())

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.translate(str.maketrans('', '', punctuation))
    text = text.lower().strip()
    text = ' '.join([i if i not in stop and i.isalpha() else '' for i in text.lower().split()])
    text = ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])
    text = re.sub(r"\s{2,}", " ", text)
    return text

train['overview'] = train['overview'].astype(str)
test['overview'] = test['overview'].astype(str)
val['overview'] = val['overview'].astype(str)

train['overview'] = train['overview'].apply(lambda text: clean_text(text))
test['overview'] = test['overview'].apply(lambda text: clean_text(text))
val['overview'] = val['overview'].apply(lambda text: clean_text(text))

dataset['overview'] = dataset['overview'].astype(str)
dataset['overview'] = dataset['overview'].apply(lambda text: clean_text(text))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Embedding, LSTM, Dropout, Dense, Input, Bidirectional, Flatten, Conv2D, MaxPooling2D, concatenate, Conv1D, MaxPooling1D
import keras.backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = dataset['overview'].map(len).max()
EMBEDDING_DIM = 300
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(dataset['overview'].values)
word_index = tokenizer.word_index


def get_embedding_matrix(typeToLoad):
    if typeToLoad == "glove":
        EMBEDDING_FILE = "/content/glove.twitter.27B.100d.txt"
        embed_size = 100
    elif typeToLoad == "word2vec":
        word2vecDict = KeyedVectors.load_word2vec_format("content/GoogleNews-vectors-negative300.bin", binary=True)
        embed_size = 300
    elif typeToLoad == "fasttext":
        EMBEDDING_FILE = "/content/wiki-news-300d-1M.vec"
        embed_size = 300

    if typeToLoad == "glove" or typeToLoad == "fasttext":
        embeddings_index = dict()
        f = open(EMBEDDING_FILE)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print("Loaded " + str(len(embeddings_index)) + " word vectors.")
    else:
        embeddings_index = dict()
        for word in word2vecDict.wv.vocab:
            embeddings_index[word] = word2vecDict.word_vec(word)
        print("Loaded " + str(len(embeddings_index)) + " word vectors.")

    embedding_matrix = 1 * np.random.randn(len(word_index) + 1, embed_size)

    embeddedCount = 0
    for word, i in word_index.items():
        i -= 1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embeddedCount += 1
    print("total embedded:", embeddedCount, "common words")

    del (embeddings_index)

    return embedding_matrix

word2vec_embedding_matrix = get_embedding_matrix("word2vec")

X_text_train = tokenizer.texts_to_sequences(train['overview'].values)
X_text_train = pad_sequences(X_text_train, maxlen=MAX_SEQUENCE_LENGTH)

X_text_test = tokenizer.texts_to_sequences(test['overview'].values)
X_text_test = pad_sequences(X_text_test, maxlen=MAX_SEQUENCE_LENGTH)

X_text_val = tokenizer.texts_to_sequences(val['overview'].values)
X_text_val = pad_sequences(X_text_val, maxlen=MAX_SEQUENCE_LENGTH)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_img_val = np.reshape(X_img_val, (X_img_val.shape[0], 3, 75, 115))
X_img_test = np.reshape(X_img_test, (X_img_test.shape[0], 3, 75, 115))
X_img_train = np.reshape(X_img_train, (X_img_train.shape[0], 3, 75, 115))

text_train_data = TensorDataset(torch.from_numpy(X_text_train), torch.from_numpy(train_labels))
img_train_data = TensorDataset(torch.from_numpy(X_img_train), torch.from_numpy(train_labels))

text_val_data = TensorDataset(torch.from_numpy(X_text_val), torch.from_numpy(val_labels))
img_val_data = TensorDataset(torch.from_numpy(X_img_val), torch.from_numpy(val_labels))

text_test_data = TensorDataset(torch.from_numpy(X_text_test), torch.from_numpy(test_labels))
img_test_data = TensorDataset(torch.from_numpy(X_img_test), torch.from_numpy(test_labels))

batch_size = 64

text_train_loader = DataLoader(text_train_data, batch_size=batch_size)
img_train_loader = DataLoader(img_train_data, batch_size=batch_size)

text_val_loader = DataLoader(text_val_data, batch_size=batch_size)
img_val_loader = DataLoader(img_val_data, batch_size=batch_size)

text_test_loader = DataLoader(text_test_data, batch_size=batch_size)
img_test_loader = DataLoader(img_test_data, batch_size=batch_size)

print(len(text_train_loader), len(img_train_loader))
print(len(text_val_loader), len(img_val_loader))
print(len(text_test_loader), len(img_test_loader))


class CNN_LSTM(nn.Module):
    def __init__(self, vocab_size, weights_matrix, n_hidden, n_layers, n_out):
        super(CNN_LSTM, self).__init__()

        # LSTM for the text overview
        self.vocab_size, self.n_hidden, self.n_out, self.n_layers = vocab_size, n_hidden, n_out, n_layers
        num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.emb.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.emb.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, self.n_hidden, self.n_layers, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.lstm_fc = nn.Linear(self.n_hidden, 128)
        # self.sigmoid = nn.Sigmoid()

        # CNN for the posters
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.cnn_dropout = nn.Dropout(0.1)
        self.cnn_fc = nn.Linear(5*2*128, 512)

        # Concat layer for the combined feature space
        # self.combined_fc1 = nn.Linear(640, 256)
        # self.combined_fc2 = nn.Linear(256, 128)
        # self.output_fc = nn.Linear(128, n_out)

        # Additive
        self.transform = nn.Linear(512, 128)
        self.combined_fc1 = nn.Linear(128, 256)
        self.combined_fc2 = nn.Linear(256, 128)
        self.output_fc = nn.Linear(128, n_out)

    def forward(self, lstm_inp, cnn_inp):
        batch_size = lstm_inp.size(0)
        hidden = self.init_hidden(batch_size)
        lstm_inp = lstm_inp.long()
        embeds = self.emb(lstm_inp)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out[:, -1])
        lstm_out = F.relu(self.lstm_fc(lstm_out))

        x = F.relu(self.conv1(cnn_inp))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool3(x)
        x = F.relu(self.conv4(x))
        x = self.max_pool4(x)
        x = x.view(-1, 5*2*128)
        x = self.cnn_dropout(x)
        cnn_out = F.relu(self.cnn_fc(x))


        combined_inp = self.transform(cnn_out)
        x_comb = F.relu(self.combined_fc1(combined_inp))
        x_comb = F.relu(self.combined_fc2(x_comb))
        out = torch.sigmoid(self.output_fc(x_comb))

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                          weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden


vocab_size = len(word_index)+1
output_size = train_labels.shape[1]
embedding_dim = 300
hidden_dim = 64
n_layers = 2
print(output_size)

model = CNN_LSTM(vocab_size, word2vec_embedding_matrix, hidden_dim, n_layers, output_size)
model.to(device)
print(model)

# lr=0.001
# # criterion = nn.MultiLabelSoftMarginLoss()
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# epochs = 200
# clip = 5

filename = "result/cnn.txt"



def movie_train(epochs,lr,criterion,optimizer,clip):
    model.train()

    for i in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for lstm, cnn in zip(text_train_loader, img_train_loader):
            lstm_inp, lstm_labels = lstm
            cnn_inp, cnn_labels = cnn
            lstm_inp, lstm_labels = lstm_inp.to(device), lstm_labels.to(device)
            cnn_inp, cnn_labels = cnn_inp.to(device), cnn_labels.to(device)
            model.zero_grad()
            output = model(lstm_inp, cnn_inp)
            loss = criterion(output.squeeze(), lstm_labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            with torch.no_grad():
                acc = torch.abs(output.squeeze() - lstm_labels.float()).view(-1)
                acc = (1. - acc.sum() / acc.size()[0])
                total_acc_train += acc
                total_loss_train += loss.item()

        train_acc = total_acc_train / len(text_train_loader)
        train_loss = total_loss_train / len(text_train_loader)
        model.eval()
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            for lstm, cnn in zip(text_val_loader, img_val_loader):
                lstm_inp, lstm_labels = lstm
                cnn_inp, cnn_labels = cnn
                lstm_inp, lstm_labels = lstm_inp.to(device), lstm_labels.to(device)
                cnn_inp, cnn_labels = cnn_inp.to(device), cnn_labels.to(device)
                model.zero_grad()
                output = model(lstm_inp, cnn_inp)
                val_loss = criterion(output.squeeze(), lstm_labels.float())
                acc = torch.abs(output.squeeze() - lstm_labels.float()).view(-1)
                acc = (1. - acc.sum() / acc.size()[0])
                total_acc_val += acc
                total_loss_val += val_loss.item()
            print("Saving model...")
            torch.save(model.state_dict(), 'content/model/pytorch_word2vec_lstm_less_dropout.pt')

        val_acc = total_acc_val / len(text_val_loader)
        val_loss = total_loss_val / len(text_val_loader)
        print(f'Epoch {i + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')

        with open(filename,'a') as f:
            f.write(f'Epoch {i + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}\n')
            f.close()

        model.train()
        torch.cuda.empty_cache()



lr=0.001
# criterion = nn.MultiLabelSoftMarginLoss()
criterion = nn.BCELoss()
weight_decay1=1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay1)

epochs = 50
clip = 5

with open(filename,'a') as f:
    f.write(f'epochs:{epochs},lr:{lr},weight decay:,{weight_decay1}\n')
    f.close()
movie_train(epochs,lr,criterion,optimizer,clip)

lr=0.01
epochs=50
weight_decay1=1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay1)
with open(filename,'a') as f:
    f.write(f'epochs:{epochs},lr:{lr},weight decay:,{weight_decay1}\n')
    f.close()
movie_train(epochs,lr,criterion,optimizer,clip)

lr=0.001
epochs=50
weight_decay1=1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay1)
with open(filename,'a') as f:
    f.write(f'epochs:{epochs},lr:{lr},weight decay:,{weight_decay1}\n')
    f.close()
movie_train(epochs,lr,criterion,optimizer,clip)

lr = 0.0005
epochs = 50
weight_decay1 = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay1)
with open(filename,'a') as f:
    f.write(f'epochs:{epochs},lr:{lr},weight decay:,{weight_decay1}\n')
    f.close()
movie_train(epochs,lr,criterion,optimizer,clip)

lr = 0.005
epochs = 50
weight_decay1 = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay1)
with open(filename,'a') as f:
    f.write(f'epochs:{epochs},lr:{lr},weight decay:,{weight_decay1}\n')
    f.close()
movie_train(epochs,lr,criterion,optimizer,clip)


lr = 0.001
epochs = 50
weight_decay1 = 1e-7
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay1)
with open(filename,'a') as f:
    f.write(f'epochs:{epochs},lr:{lr},weight decay:,{weight_decay1}\n')
    f.close()
movie_train(epochs,lr,criterion,optimizer,clip)

lr = 0.001
epochs = 50
weight_decay1 = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay1)
with open(filename,'a') as f:
    f.write(f'epochs:{epochs},lr:{lr},weight decay:,{weight_decay1}\n')
    f.close()
movie_train(epochs,lr,criterion,optimizer,clip)