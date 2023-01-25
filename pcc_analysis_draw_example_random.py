# 除法总是会返回真实的商，不管操作数是整形还是浮点型。执行from __future__ import division 指令就可以做到这一点。
from  __future__ import  division
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from collections import OrderedDict
import torch.nn as nn
from networks.uap import UAP
from utils.data import get_data_specs, get_data
from utils.utils import softmax, get_imagenet_dicts
from utils.network import get_network


MODEL_ARCH = "vgg19"
TARGET_CLASS = 150
TUAP_RESULT_PATH = "./vgg19_sea_lion"

NGPU=1
USE_CUDA=True
idx2label, _ = get_imagenet_dicts()
props = dict(facecolor='none', edgecolor='black', pad=5, alpha=0.5)

num_classes, (mean, std), input_size, num_channels = get_data_specs("imagenet")
std_tensor = torch.tensor(std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
mean_tensor = torch.tensor(mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
if USE_CUDA:
    std_tensor = std_tensor.cuda()
    mean_tensor = mean_tensor.cuda()
# 使用ImageNet中的val数据作为测试数据，应该是这个数据集比较小，只有5G吧，训练数据集太大了129G。
_, data_test = get_data("imagenet", pretrained_dataset="imagenet")
data_test_loader = torch.utils.data.DataLoader(data_test,
                                                batch_size=1,
                                                shuffle=True,
                                                num_workers=0,
                                                pin_memory=True)
test_iter = iter(data_test_loader)
# 实现获取目标模型的代码，是从torchvision中下载的vgg19
target_model = get_network(MODEL_ARCH, input_size=input_size, num_classes=num_classes)
target_model = torch.nn.DataParallel(target_model, device_ids=list(range(NGPU)))


def plot(img1, img1_logit,
         img2, img2_logit,
         img1_img2, img1_img2_logit,
         center_label):
    get_imagenet_dicts()

    fig = plt.figure(figsize=(27, 6))

    ax = plt.subplot(1, 5, 1)
    ax.imshow(img1, vmin=0, vmax=1)
    cl_img_1 = np.argmax(img1_logit)
    img1_label = idx2label[cl_img_1].replace("_", " ")
    # ax.set_xlabel("{} ({:.2f}%)".format("Original image:"+img1_label, softmax(img1_logit[0])[cl_img_1] * 100))

    ax.set_xlabel("{}".format("Original image:"+img1_label))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(1, 5, 2)
    ax.imshow(img2, vmin=0, vmax=1)
    cl_img_2 = np.argmax(img2_logit)
    img2_label = idx2label[cl_img_2].replace("_", " ")
    ax.set_xlabel("{}".format("Superimposd image:"+img2_label))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(1, 5, 3)
    ax.imshow(img1_img2, vmin=0, vmax=1)
    cl_img_3 = np.argmax(img1_img2_logit)
    img3_label = idx2label[cl_img_3].replace("_", " ")
    # ax.set_xlabel("{} ({:.2f}%)".format("Predicted:"+img3_label, softmax(img1_img2_logit[0])[cl_img_3] * 100))
    ax.set_xlabel("Synthetic image")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    return fig

img1, lbl1 = next(test_iter)
if USE_CUDA:
    img1 = img1.cuda()
    lbl1 = lbl1.cuda()
# Getting logits
logit1 = target_model(img1)

cl1 = torch.argmax(logit1)

# 在原始图片上加上随机图片对比。
img2, _ = next(test_iter)
noise_img = img2.cuda()
### Perpearing the inputs
img1_logit = target_model(img1).cpu().detach().numpy()
plot_img1 = img1[0].cpu().detach().numpy()
plot_img1 = np.transpose(plot_img1, (1, 2, 0))
plot_img1 = plot_img1 * std + mean

# Original img + Other image
noise_img_logit = target_model((noise_img - mean_tensor)/std_tensor).cpu().detach().numpy()
plot_noise_img = noise_img[0].cpu().detach().numpy()
plot_noise_img = np.transpose(plot_noise_img, (1, 2, 0)) + 0.5

img_noise_img = ((img1 * std_tensor + mean_tensor + noise_img) - mean_tensor)/std_tensor
img_noise_img_logit = target_model(img_noise_img).cpu().detach().numpy()
plot_img_noise_img = np.clip((plot_img1 + plot_noise_img - 0.5), 0, 1)

fig = plot(plot_img1, img1_logit, plot_noise_img, noise_img_logit, plot_img_noise_img, img_noise_img_logit, center_label="Noise")
fig.savefig("original_img_random.png")
exit()