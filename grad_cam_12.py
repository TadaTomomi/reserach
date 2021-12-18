import cv2
import glob
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

sex_label = ['male', 'female']
age_label = ['~14', '15~19', '20~24', '25~29', '30~34', '35~39', '40~']

def make_data(data_directory):
  max = 96
  datasets = []
  volume = []
  files = sorted(glob.glob(data_directory + "/*.jpg"))
  num = int(np.ceil(len(files) / 3))
  for myFile in files[::3]:
    image1 = cv2.imread(myFile)
    image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #圧縮
    image3 = cv2.resize(image2, (96, 96))
    image4 = np.array([image3])
    volume.append(image4)
  zero = np.zeros(((max-num), 1, 96, 96))
  np_volume = np.array(volume)
  # volume = np.append(np_volume, zero, axis=0).tolist()
  volume = np.append(np_volume, zero, axis=0) / 255.0
  volume = volume.tolist()
  datasets.append(volume)
  torch_datasets = torch.tensor(datasets).permute(0, 2, 1, 3, 4)

  return torch_datasets.float()

# img = make_data("/home/student/datasets/CT393/valid/0321.27y6m.f")
# print(img.shape)

def make_3d(data_directory):
  max = 96
  volume = []
  files = sorted(glob.glob(data_directory + "/*.jpg"))
  num = int(np.ceil(len(files) / 3))
  for myFile in files[::3]:
    image = cv2.imread(myFile)
    volume.append(image)
  zero = np.zeros(((max-num), 512, 512, 3))
  np_volume = np.array(volume)
  # volume = np.append(np_volume, zero, axis=0).tolist()
  volume = np.append(np_volume, zero, axis=0)
  volume = volume.tolist()
  return np.array(volume)

# img_3d = make_3d("/home/student/datasets/CT393/valid/0361.18y10m.f")
# print(img_3d.shape)

# バイリニア補間法でリサイズ
def resize_bilinear(src, dd, hd, wd):

    # 出力画像用の配列生成（要素は全て空）
    dst = np.empty((dd, hd, wd))

    # 元画像のサイズを取得
    d, h, w = src.shape[0], src.shape[1], src.shape[2]

    # 拡大率を計算
    ax = wd / float(w)
    ay = hd / float(h)
    az = dd / float(d)

    # バイリニア補間法
    for zd in range(0, dd):
      for yd in range(0, hd):
          for xd in range(0, wd):
              x, y, z = xd/ax, yd/ay, zd/az
              ox, oy, oz = int(x), int(y), int(z)

              # 存在しない座標の処理
              if ox > w - 2:
                  ox = w - 2
              if oy > h - 2:
                  oy = h - 2
              if oz > d - 2:
                  oz = d - 2

              # 重みの計算
              dx = x - ox
              dy = y - oy
              dz = z - oz

              # 出力画像の画素値を計算 #####################後で
              dst[zd][yd][xd] = (1-dx) * (1-dy) * (1-dz) * src[oz][oy][ox] + \
                  dx * (1-dy) * (1-dz) * src[oz][oy][ox+1] + (1-dx) * dy * (1-dz) * src[oz][oy+1][ox] + \
                  dx * dy * (1-dz) * src[oz][oy+1][ox+1] + (1-dx) * (1-dy) * dz * src[oz+1][oy][ox] +  \
                  dx * (1-dy) * dz * src[oz+1][oy][ox+1] + (1-dx) * dy * dz * src[oz+1][oy+1][ox] + \
                  dx * dy * dz * src[oz+1][oy+1][ox+1]

    return dst

# 最近傍補間法でリサイズ
def resize_nearest(src,d,h,w):
    # 出力画像用の配列生成（要素は全て空）
    dst = np.empty((d,h,w))

    # 元画像のサイズを取得
    di, hi, wi = src.shape[0], src.shape[1], src.shape[2]

    # 拡大率を計算
    ax = w / float(wi)
    ay = h / float(hi)
    az = d / float(di)

    # 最近傍補間
    for z in range(0, d):
      for y in range(0, h):
        for x in range(0, w):
          xi, yi, zi = int(round(x/ax)), int(round(y/ay)), int(round(z/az))
          # 存在しない座標の処理
          if xi > wi -1: xi = wi -1
          if yi > hi -1: yi = hi -1
          if zi > di -1: zi = di -1

          dst[z][y][x] = src[zi][yi][xi]

    return dst


def gradcam(net, img_fpath):

    net.eval()

    img = make_data(img_fpath)
    transform = transforms.Compose([
        transforms.Normalize(0.07, 0.14)
    ])
    
    img = transform(img)
    img = img.to(device)
    # img = img.unsqueeze(0)

    # get features from the last convolutional layer
    x = net.features(img)
    feature = x

    # hook for the gradients
    def __extract_grad(grad):
        global feature_grad
        feature_grad = grad
    feature.register_hook(__extract_grad)

    # get the output from the whole VGG architecture
    x = x.view(x.size(0), -1)
    x = net.fc1(x)
    x = net.dropout(x)
    x = net.fc2(x)
    sex = net.fc_sex(x)
    age = net.fc_age(x)
    pred_sex = torch.argmax(sex).item()
    pred_age = torch.argmax(age).item()
    print(sex_label[pred_sex])
    print(age_label[pred_age])

    # get the gradient of the output
    # output = sex[:, pred_sex] + age[:, pred_age]
    # output = (sex[:, pred_sex] * 7 + age[:, pred_age] * 2) / 9
    #別々にバックワード
    output = sex[:, pred_sex]
    # output = age[:, pred_age]
    output.backward()

    # pool the gradients across the channels
    pooled_grad = torch.mean(feature_grad, dim=[0, 2, 3, 4])

    # weight the channels with the corresponding gradients
    # (L_Grad-CAM = alpha * A)
    feature = feature.detach()  #####(1, 128, 12, 12, 12)
    for i in range(feature.shape[1]):
        feature[:, i, :, :, :] *= pooled_grad[i] 

    # average the channels and create an heatmap
    # ReLU(L_Grad-CAM)
    heatmap = torch.mean(feature, dim=1).squeeze() ####(12, 12, 12)
    heatmap = heatmap.cpu()
    heatmap = np.maximum(heatmap, 0) ##マイナスを0にする

    # normalization for plotting
    heatmap = heatmap / torch.max(heatmap)
    heatmap = heatmap.numpy()

    # project heatmap onto the input image
    img_3d = make_3d(img_fpath)
    heatmap = resize_bilinear(heatmap, 96, 512, 512)
    heatmap = np.uint8(255 * heatmap)
    plt.figure(figsize=(32,32))
    for i in range(12):
      num = [3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91] 
      heatmap_2d = cv2.applyColorMap(heatmap[num[i]], cv2.COLORMAP_JET)
      superimposed_img = heatmap_2d * 0.4 + img_3d[num[i]]
      superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
      superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
      
      plt.subplot(3, 4, i+1)
      plt.title(f"{num[i]+1}")
      plt.imshow(superimposed_img)
      plt.axis('off')

    plt.savefig("CT/heatmap_12_sex.jpg")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

model = torch.load('model.pth')
model.to(device)
directory = "/home/student/datasets/CT393/valid/0393.23y8m.m"
gradcam(model, directory)