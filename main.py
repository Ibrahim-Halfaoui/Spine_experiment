import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorboardX import SummaryWriter
import os
dir_ = os.getcwd()

## Dataset Preparation
class Dataset_spine(object):
        def __init__(self, csv_path):
            self.csv_path = csv_path
            self.df = pd.read_csv(self.csv_path)
            self.df.head()
            sns.countplot(x='Class_att', data=self.df)
            self.df['Class_att'] = self.df['Class_att'].astype('category')
            encode_map = {
                'Abnormal': 1,
                'Normal': 0
            }
            self.df['Class_att'].replace(encode_map, inplace=True)
            X = self.df.iloc[:, 0:-2]
            y = self.df.iloc[:, -2]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=69)
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

## train data
class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


## test data
class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)

## Model Arch
class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(12, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

## Accuracy Computation
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

## Training function
def train_model(model, device, snapshot_dir, train_loader, val_loader):
    print('Start training...\n')
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if not os.path.isdir(snapshot_dir):
        os.mkdir(snapshot_dir)
    writer = SummaryWriter(snapshot_dir)

    model.train()
    # Train loop
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(f'Epoch {e + 0:03}: | TrainLoss: {epoch_loss / len(train_loader):.5f} | trainAcc: {epoch_acc / len(train_loader):.3f}')
        writer.add_scalar('TrainLoss', epoch_loss / len(train_loader), e)
        writer.add_scalar('TrainAcc', epoch_acc / len(train_loader), e)

        # Validation loop
        epoch_loss_t = 0
        epoch_acc_t = 0
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            epoch_loss_t += loss.item()
            epoch_acc_t += acc.item()
        print(f'Epoch {e + 0:03}: | ValLoss: {epoch_loss / len(val_loader):.5f} | ValAcc: {epoch_acc / len(val_loader):.3f}')
        writer.add_scalar('ValLoss', epoch_loss / len(val_loader), e)
        writer.add_scalar('ValAcc', epoch_acc / len(val_loader), e)

        # Saving snapshots
        torch.save(model.state_dict(), snapshot_dir + '/epoch_{}.pt'.format(e))

## Test function
def test_model(model):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    return y_pred_list

## Main
if __name__ == "__main__":
    # parse Arguments
    parser = argparse.ArgumentParser(description='Model for tabular data clasiification.')
    parser.add_argument('--mode', type=str, help='train or test', default='test')
    parser.add_argument('--data_path', type=str, help='path to the data',
                        default='dataset_spine.csv')
    parser.add_argument('--batch_size', type=int, help='batch size',
                        default=64)  
    parser.add_argument('--num_epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('--lr_rate', type=float, help='initial learning rate', default=1e-3)
    parser.add_argument('--log_dir', default='/log/', type=str,   help='directory to save checkpoints and summaries')
    parser.add_argument('--snapshot',  type=str,   help='path to a specific checkpoint to load', default='')
    args = parser.parse_args()
    csv_path = args.data_path
    EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr_rate
    experiment_name = '/tabular_' + str(args.lr_rate) + '_'+ str(args.batch_size) + '_' + str(args.num_epochs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Prepare Data
    data = Dataset_spine(csv_path)
    train_data = trainData(torch.FloatTensor(data.X_train), torch.FloatTensor(data.y_train))
    test_data = testData(torch.FloatTensor(data.X_test))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    
    # Run training
    model = binaryClassification()
    train_model(model, device, dir_ + experiment_name, train_loader, train_loader)
  

