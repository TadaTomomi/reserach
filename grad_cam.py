import cv2
import glob
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def make_data(data_directory):
  max = 100
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

def gradcam(net, img_fpath):

    net.eval()

    img = make_data(img_fpath)
    transform = transforms.Compose([
        transforms.Normalize(0.07, 0.14)
    ])
    
    img = transform(img)
    # img = img.unsqueeze(0)

    # get features from the last convolutional layer
    x = net.features[:10](img)
    features = x

    # hook for the gradients
    def __extract_grad(grad):
        global feature_grad
        feature_grad = grad
    features.register_hook(__extract_grad)

    # get the output from the whole VGG architecture
    x = x.view(x.size(0), -1)
    x = net.fc1(x)
    x = net.dropout(x)
    x = net.fc2(x)
    sex = net.fc_sex(x)
    age = net.fc_age(x)
    pred_sex = torch.argmax(sex).item()
    pred_age = torch.argmax(age).item()
    print(pred_sex)
    print(pred_age)

    # get the gradient of the output
    output = sex[:, pred_sex] + age[:, pred_age]############
    output.backward() #################################

    # pool the gradients across the channels
    pooled_grad = torch.mean(feature_grad, dim=[0, 2, 3, 4])############

    # weight the channels with the corresponding gradients
    # (L_Grad-CAM = alpha * A)
    features = features.detach()
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_grad[i] 

    # average the channels and create an heatmap
    # ReLU(L_Grad-CAM)
    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)

    # normalization for plotting
    heatmap = heatmap / torch.max(heatmap)
    heatmap = heatmap.numpy()

    print(feature_grad.shape)
    print(heatmap.shape)

    # project heatmap onto the input image
    # img = cv2.imread(img_fpath)
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed_img = heatmap * 0.4 + img
    # superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
    # superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    # plt.imshow(superimposed_img)
    # plt.show()

    img = cv2.imread(img_fpath + "/IM-0001-0151.jpg")
    heatmap = cv2.resize(heatmap[5], (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    # plt.imshow(superimposed_img)
    # plt.show()
    plt.imsave("CT/heatmap.jpg", superimposed_img)


model = torch.load('model.pth').cpu()
directory = "/home/student/datasets/CT393/valid/0342.16y5m.f"
gradcam(model, directory)