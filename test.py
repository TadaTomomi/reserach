import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from make_data import make_data
from make_label import make_label
from datasets import CTDataset
from model import CNN3D
from train import train
from valid import valid
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

#3次元CT(3次元配列)をnumpy配列に入れる
valid_data = make_data("/home/student/datasets/CT200_160/valid/*")

#ラベルをnumpy配列に入れる
valid_label = make_label('/home/student/datasets/CT200_160/valid_label.csv')

#前処理を定義
valid_transform = transforms.Compose([
    transforms.Normalize((0.5, ), (0.5, ))
    ])


#データセットを定義
valid_dataset = CTDataset(valid_data, valid_label, transform=valid_transform)

#データローダーを定義
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False)

#モデルを読み込む
model = torch.load('model.pth')
model.to(device)

#クラスラベル
sex_label = ['male', 'female']
age_label = ['~14', '15~19', '20~24', '25~29', '30~34', '35~39', '40~']

#4つ試してみる
dataiter = iter(valid_dataloader)
X, y = dataiter.next()
print(X[1][0][50][30][30])
X, y = X.to(device), y.to(device)
# print(X)
# print(X.shape)
# print(y)
# print(y.shape)
y_sex = y[:, 0]
y_age = y[:, 1]
print('GroundTruth: ', ' '.join('%5s' % sex_label[y_sex[j]] for j in range(4)))
print('GroundTruth: ', ' '.join('%5s' % age_label[y_age[j]] for j in range(4)))
pred_sex, pred_age = model(X)
print(pred_sex)
print(pred_age)
_, pred_sex = torch.max(pred_sex, 1)
_, pred_age = torch.max(pred_age, 1)
# print(pred_sex)
# print(pred_age)
print('Predicted: ', ' '.join('%5s' % sex_label[pred_sex[j]] for j in range(4)))
print('Predicted: ', ' '.join('%5s' % age_label[pred_age[j]] for j in range(4)))

np_y_sex = y_sex.to('cpu').detach().numpy().copy()
np_pred_sex = pred_sex.to('cpu').detach().numpy().copy()
np_y_age = y_age.to('cpu').detach().numpy().copy()
np_pred_age = pred_age.to('cpu').detach().numpy().copy()
sex_label = [0, 1]
age_label = [0, 1, 2, 3, 4, 5, 6]
cm_sex = confusion_matrix(np_y_sex, np_pred_sex, labels = sex_label)
cm_age = confusion_matrix(np_y_age, np_pred_age, labels = age_label)
print(cm_sex)
print(cm_age)

#テスト
size = len(test_dataloader.dataset)
model.eval()
correct_sex, correct_age = 0.0, 0.0
pred_sex_list = []
pred_age_list = []
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred_sex, pred_age = model(X)
            
        #yをsexとageに分ける
        y_sex = y[:, 0]
        y_age = y[:, 1]
            
        correct_sex += (y_sex == pred_sex.argmax(1)).type(torch.float).sum().item()
        correct_age += (y_age == pred_age.argmax(1)).type(torch.float).sum().item()

        _, pred_sex = torch.max(pred_sex, 1)
        _, pred_age = torch.max(pred_age, 1)

        pred_sex = pred_sex.to('cpu').detach().numpy().copy()
        pred_age = pred_age.to('cpu').detach().numpy().copy()

        pred_sex_list = np.concatenate([pred_sex_list, pred_sex])
        pred_age_list = np.concatenate([pred_age_list, pred_age])

correct_sex /= size
correct_age /= size
correct_sex *= 100
correct_age *= 100

print(f"Accuracy(sex): {(correct_sex):>0.1f}%")
print(f"Accuracy(age): {(correct_age):>0.1f}% \n")

np_y_sex = valid_label[0].detach().numpy().copy()
np_y_age = valid_label[1].detach().numpy().copy()

print(np_y_sex)
print(np_y_age)
print(pred_sex_list)
print(pred_age_list)

cm_sex = confusion_matrix(np_y_sex, pred_sex_list, labels = sex_label)
cm_age = confusion_matrix(np_y_age, pred_age_list, labels = age_label)
print(cm_sex)
print(cm_age)