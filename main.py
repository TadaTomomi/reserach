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

#3次元CT(3次元配列)を配列に入れる
train_data = make_data("/home/student/datasets/CT200_160/train/*")
valid_data = make_data("/home/student/datasets/CT200_160/valid/*")

#ラベルを配列に入れる
train_label = make_label('/home/student/datasets/CT200_160/train_label.csv')
valid_label = make_label('/home/student/datasets/CT200_160/valid_label.csv')

#前処理を定義
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize((0.5, ), (0.5, ))
    ])

valid_transform = transforms.Compose([
    transforms.Normalize((0.5, ), (0.5, ))
    ])


#データセットを定義
train_dataset = CTDataset(train_data, train_label, transform=train_transform)
valid_dataset = CTDataset(valid_data, valid_label, transform=valid_transform)

#データローダーを定義
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False)

#モデルを読み込む
model = CNN3D().to(device)

print(model)

#損失関数
sex_weight = torch.tensor([7.0, 3.0]).cuda()
age_weight = torch.tensor([12.0, 2.0, 3.0, 4.0, 6.0, 6.0, 12.0]).cuda()
sex_criterion = nn.CrossEntropyLoss(weight=sex_weight)
age_criterion = nn.CrossEntropyLoss(weight=age_weight)

# sex_criterion = nn.CrossEntropyLoss()
# age_criterion = nn.CrossEntropyLoss()

#最適化手法
optimizer = optim.Adam(model.parameters(), lr=0.01)

#学習・検証
epochs = 10
train_loss_list = []
valid_loss_list = []
train_correct_sex_list = []
valid_correct_sex_list = []
train_correct_age_list = []
valid_correct_age_list = []
train_loss_sex_list = []
valid_loss_sex_list = []
train_loss_age_list = []
valid_loss_age_list = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_correct_sex, train_correct_age = train(train_dataloader, model, sex_criterion, age_criterion, optimizer, device)
    valid_loss, valid_correct_sex, valid_correct_age = valid(valid_dataloader, model, sex_criterion, age_criterion, device)
    train_loss_list.append(train_loss[0])
    valid_loss_list.append(valid_loss[0])
    train_correct_sex_list.append(train_correct_sex)
    valid_correct_sex_list.append(valid_correct_sex)
    train_correct_age_list.append(train_correct_age)
    valid_correct_age_list.append(valid_correct_age)
    train_loss_sex_list.append(train_loss[1])
    valid_loss_sex_list.append(valid_loss[1])
    train_loss_age_list.append(train_loss[2])
    valid_loss_age_list.append(valid_loss[2])
print("Done!")

#モデルの保存
torch.save(model, 'model.pth')

# 損失の合計グラフ表示
fig1 = plt.figure()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(train_loss_list, label='train loss')
plt.plot(valid_loss_list, label='validation loss')
plt.xlim()
plt.legend()
fig1.savefig("graph/loss.png")

#性別の正答率のグラフ表示
fig2 = plt.figure()
plt.xlabel("epoch")
plt.ylabel("Sex Accuracy")
plt.plot(train_correct_sex_list, label='train Accuracy')
plt.plot(valid_correct_sex_list, label='validation Accuracy')
plt.ylim(0, 100)
plt.legend()
fig2.savefig("graph/sex.png")

#年齢の正答率のグラフ表示
fig3 = plt.figure()
plt.xlabel("epoch")
plt.ylabel("Age Accuracy")
plt.plot(train_correct_age_list, label='train Accuracy')
plt.plot(valid_correct_age_list, label='validation Accuracy')
plt.ylim(0, 100)
plt.legend()
fig3.savefig("graph/age.png")

# 性別の損失のグラフ表示
fig4 = plt.figure()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(train_loss_sex_list, label='train loss')
plt.plot(valid_loss_sex_list, label='validation loss')
plt.xlim()
plt.legend()
fig4.savefig("graph/loss_sex.png")
# 年齢の損失のグラフ表示
fig5 = plt.figure()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(train_loss_age_list, label='train loss')
plt.plot(valid_loss_age_list, label='validation loss')
plt.xlim()
plt.legend()
fig5.savefig("graph/loss_age.png")

#ラベルgi
sex_label = ['male', 'female']
age_label = ['~14', '15~19', '20~24', '25~29', '30~34', '35~39', '40~']

#バッチ1つ試してみる
dataiter = iter(valid_dataloader)
X, y = dataiter.next()
X, y = X.to(device), y.to(device)
# print(X.shape)
# print(y)
# print(y.shape)
y_sex = y[:, 0]
y_age = y[:, 1]
print('GroundTruth: ', ' '.join('%5s' % sex_label[y_sex[j]] for j in range(4)))
print('GroundTruth: ', ' '.join('%5s' % age_label[y_age[j]] for j in range(4)))
pred_sex, pred_age = model(X)
# print(pred_sex)
# print(pred_age)
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