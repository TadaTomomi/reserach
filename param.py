import torch
import numpy as np
import matplotlib.pyplot as plt

#モデルを読み込む
model = torch.load('model.pth').cpu()

stdict = model.state_dict()
kernel = np.array(stdict['features.0.weight'][0][0][0])
# print(kernel)
plt.imshow(kernel.reshape(5, 5), cmap='gray')
plt.show()
# print(list(model.parameters()))