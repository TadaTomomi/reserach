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
  volume = np.append(np_volume, zero, axis=0)
  volume = volume.tolist()
  return np.array(volume)


def gradcam(net, img_fpath):

    net.eval()

    img = make_data(img_fpath)
    transform = transforms.Compose([
        transforms.Normalize(0.07, 0.14)
    ])
    
    img = transform(img)
    img = img.to(device)

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
    # print(sex)
    # print(age)
    print(sex_label[pred_sex])
    print(age_label[pred_age])

    # get the gradient of the output    
    # output = (sex[:, pred_sex] + age[:, pred_age]) / 2
    # output = (sex[:, pred_sex] * 7 + age[:, pred_age] * 2) / 9
    #別々にバックワード
    # output = sex[:, pred_sex]
    output = age[:, pred_age]
    output.backward()

    # pool the gradients across the channels
    pooled_grad = torch.mean(feature_grad, dim=[0, 2, 3, 4])

    # weight the channels with the corresponding gradients
    # (L_Grad-CAM = alpha * A)
    feature = feature.detach()
    for i in range(feature.shape[1]):
        feature[:, i, :, :] *= pooled_grad[i] 

    # average the channels and create an heatmap
    # ReLU(L_Grad-CAM)
    heatmap = torch.mean(feature, dim=1).squeeze()
    heatmap = heatmap.cpu()
    heatmap = np.maximum(heatmap, 0)

    # normalization for plotting
    heatmap = heatmap / torch.max(heatmap)
    heatmap = heatmap.numpy()

    # print(feature_grad.shape)
    # print(heatmap.shape)

    # project heatmap onto the input image
    data_3d = make_3d(img_fpath)
    plt.figure(figsize=(16,16))
    for i in range(12):
      num = [3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91]
      # img = cv2.imread(img_fpath + "/IM-0001-0"+num[i]+".jpg")
      img = data_3d[num[i]]
      heatmap_2d = cv2.resize(heatmap[i], (img.shape[1], img.shape[0]))
      heatmap_2d = np.uint8(255 * heatmap_2d)
      heatmap_2d = cv2.applyColorMap(heatmap_2d, cv2.COLORMAP_JET)
      superimposed_img = heatmap_2d * 0.4 + img
      superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
      superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
      
      plt.subplot(3, 4, i+1)
      plt.title(f"{num[i]+1}")
      plt.imshow(superimposed_img)
      plt.axis('off')

    plt.savefig("CT/heatmap_age.jpg")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

model = torch.load('model.pth')
model.to(device)
directory = "/home/student/datasets/CT393/valid/0393.23y8m.m"
gradcam(model, directory)