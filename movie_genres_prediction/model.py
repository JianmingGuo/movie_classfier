import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN_LSTM_ADD(nn.Module):
    def __init__(self, vocab_size, weights_matrix, n_hidden, n_layers, n_out):
        super(CNN_LSTM_ADD, self).__init__()

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

        # combined_inp = torch.cat((cnn_out, lstm_out), 1)
        combined_inp = lstm_out + self.transform(cnn_out)
        # combined_inp = self.transform(cnn_out)
        # combined_inp = lstm_out
        x_comb = F.relu(self.combined_fc1(combined_inp))
        x_comb = F.relu(self.combined_fc2(x_comb))
        out = torch.sigmoid(self.output_fc(x_comb))

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                          weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden

class CNN_LSTM_concat(nn.Module):
    def __init__(self, vocab_size, weights_matrix, n_hidden, n_layers, n_out):
        super(CNN_LSTM_concat, self).__init__()

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
        self.combined_fc1 = nn.Linear(640, 256)
        self.combined_fc2 = nn.Linear(256, 128)
        self.output_fc = nn.Linear(128, n_out)

        # Additive
        # self.transform = nn.Linear(512, 128)
        # self.combined_fc1 = nn.Linear(128, 256)
        # self.combined_fc2 = nn.Linear(256, 128)
        # self.output_fc = nn.Linear(128, n_out)

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

        combined_inp = torch.cat((cnn_out, lstm_out), 1)
        x_comb = F.relu(self.combined_fc1(combined_inp))
        x_comb = F.relu(self.combined_fc2(x_comb))
        out = torch.sigmoid(self.output_fc(x_comb))

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                          weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden

class CNN(nn.Module):
    def __init__(self, vocab_size, weights_matrix, n_hidden, n_layers, n_out):
        super(CNN, self).__init__()

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

class LSTM(nn.Module):
    def __init__(self, vocab_size, weights_matrix, n_hidden, n_layers, n_out):
        super(LSTM, self).__init__()

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

 
        combined_inp = lstm_out
        x_comb = F.relu(self.combined_fc1(combined_inp))
        x_comb = F.relu(self.combined_fc2(x_comb))
        out = torch.sigmoid(self.output_fc(x_comb))

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                          weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden

class RESNET_LSTM_ADD(nn.Module):
    def __init__(self, vocab_size, weights_matrix, n_hidden, n_layers, n_out):
        super(CNN_LSTM_ADD, self).__init__()

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

        # resnet for the posters
        self.backbone = resnet18(pretrained=True)

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

        cnn_out = self.backbone(cnn_inp)

        # combined_inp = torch.cat((cnn_out, lstm_out), 1)
        combined_inp = lstm_out + self.transform(cnn_out)
        # combined_inp = self.transform(cnn_out)
        # combined_inp = lstm_out
        x_comb = F.relu(self.combined_fc1(combined_inp))
        x_comb = F.relu(self.combined_fc2(x_comb))
        out = torch.sigmoid(self.output_fc(x_comb))

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                          weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden

class RESNET_LSTM_concat(nn.Module):
    def __init__(self, vocab_size, weights_matrix, n_hidden, n_layers, n_out):
        super(CNN_LSTM_concat, self).__init__()

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

        # resnet for the posters
        self.backbone = resnet18(pretrained=True)

        # Concat layer for the combined feature space
        self.combined_fc1 = nn.Linear(640, 256)
        self.combined_fc2 = nn.Linear(256, 128)
        self.output_fc = nn.Linear(128, n_out)

        # Additive
        # self.transform = nn.Linear(512, 128)
        # self.combined_fc1 = nn.Linear(128, 256)
        # self.combined_fc2 = nn.Linear(256, 128)
        # self.output_fc = nn.Linear(128, n_out)

    def forward(self, lstm_inp, cnn_inp):
        batch_size = lstm_inp.size(0)
        hidden = self.init_hidden(batch_size)
        lstm_inp = lstm_inp.long()
        embeds = self.emb(lstm_inp)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out[:, -1])
        lstm_out = F.relu(self.lstm_fc(lstm_out))

        cnn_out = self.backbone(cnn_inp)

        combined_inp = torch.cat((cnn_out, lstm_out), 1)
        # combined_inp = lstm_out + self.transform(cnn_out)
        # combined_inp = self.transform(cnn_out)
        # combined_inp = lstm_out
        x_comb = F.relu(self.combined_fc1(combined_inp))
        x_comb = F.relu(self.combined_fc2(x_comb))
        out = torch.sigmoid(self.output_fc(x_comb))

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                          weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden

class RESNET(nn.Module):
    def __init__(self, vocab_size, weights_matrix, n_hidden, n_layers, n_out):
        super(RESNET, self).__init__()

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

        # resnet for the posters
        self.backbone = resnet18(pretrained=True)

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

        cnn_out = self.backbone(cnn_inp)

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

