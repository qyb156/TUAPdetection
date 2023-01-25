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

# 从历史训练模型中load出来。
tuap_generator = UAP(shape=(input_size, input_size),
                        num_channels=num_channels,
                        mean=mean,
                        std=std,
                        use_cuda=USE_CUDA)
model_weights_path = os.path.join(TUAP_RESULT_PATH, "checkpoint.pth.tar")
network_data = torch.load(model_weights_path)
tuap_generator.load_state_dict(network_data['state_dict'])
# 用OrderedDict会根据放入元素的先后顺序进行排序。所以输出的值是排好序的
tuap_perturbed_net = nn.Sequential(OrderedDict([('generator', tuap_generator), ('target_model', target_model)]))
tuap_perturbed_net = torch.nn.DataParallel(tuap_perturbed_net, device_ids=list(range(NGPU)))

tuap_perturbed_net.module.target_model.eval()
tuap_perturbed_net.module.generator.eval()

if USE_CUDA:
    tuap_generator.cuda()
    tuap_perturbed_net.cuda()


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
    ax.set_xlabel("{} ({:.2f}%)".format(img1_label, softmax(img1_logit[0])[cl_img_1] * 100))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(1, 5, 2)
    ax.imshow(img2, vmin=0, vmax=1)
    cl_img_2 = np.argmax(img2_logit)
    img2_label = idx2label[cl_img_2].replace("_", " ")
    ax.set_xlabel("{}".format(center_label))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(1, 5, 3)
    ax.imshow(img1_img2, vmin=0, vmax=1)
    cl_img_3 = np.argmax(img1_img2_logit)
    img3_label = idx2label[cl_img_3].replace("_", " ")
    ax.set_xlabel("{} ({:.2f}%)".format(img3_label, softmax(img1_img2_logit[0])[cl_img_3] * 100))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Lgit plots
    ax = plt.subplot(1, 5, 4)
    ax.scatter(img1_logit[0], img1_img2_logit[0], s=2, color="blue")
    ax.set_xlabel("$L_a$")
    ax.set_ylabel("$L_c$", rotation=0)

    # Calculating the covariance matrices
    logit_mat1 = np.concatenate((img1_logit, img1_img2_logit), axis=0)
    origin1 = np.mean(logit_mat1, axis=1)

    # Calculate Pearson coefficient
    pcc1 = np.corrcoef(logit_mat1)
    props = dict(facecolor='none', edgecolor='black', pad=5, alpha=0.5)
    ax.text(0.65, 0.05, "PCC: {:.2f}".format(pcc1[0, 1]), transform=ax.transAxes, fontsize=20,
            verticalalignment='bottom', bbox=props)

    ax = plt.subplot(1, 5, 5)
    ax.scatter(img2_logit[0], img1_img2_logit[0], s=2, color="red")
    ax.set_xlabel("$L_b$")
    ax.set_ylabel("$L_c$", rotation=0)

    # Calculating the covariance matrices
    logit_mat2 = np.concatenate((img2_logit, img1_img2_logit), axis=0)

    # Calculate Pearson coefficient
    pcc2 = np.corrcoef(logit_mat2)
    ax.text(0.65, 0.05, "PCC: {:.2f}".format(pcc2[0, 1]), transform=ax.transAxes, fontsize=20,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    return fig

img1, lbl1 = next(test_iter)
if USE_CUDA:
    img1 = img1.cuda()
    lbl1 = lbl1.cuda()
# Getting logits
logit1 = target_model(img1)
logit2 = tuap_perturbed_net(img1)

cl1 = torch.argmax(logit1)
cl2 = torch.argmax(logit2)

# 在原始图片上加上高斯噪音进行pcc对比。
# # Img + Gaussian Noise Image
# noise_img = torch.randn_like(img1) * 0.2
#
# ### Perpearing the inputs
# img1_logit = target_model(img1).cpu().detach().numpy()
# plot_img1 = img1[0].cpu().detach().numpy()
# plot_img1 = np.transpose(plot_img1, (1, 2, 0))
# plot_img1 = plot_img1 * std + mean
#
# # Original img + Other image
# noise_img_logit = target_model((noise_img - mean_tensor)/std_tensor).cpu().detach().numpy()
# plot_noise_img = noise_img[0].cpu().detach().numpy()
# plot_noise_img = np.transpose(plot_noise_img, (1, 2, 0)) + 0.5
#
# img_noise_img = ((img1 * std_tensor + mean_tensor + noise_img) - mean_tensor)/std_tensor
# img_noise_img_logit = target_model(img_noise_img).cpu().detach().numpy()
# plot_img_noise_img = np.clip((plot_img1 + plot_noise_img - 0.5), 0, 1)
#
# fig = plot(plot_img1, img1_logit, plot_noise_img, noise_img_logit, plot_img_noise_img, img_noise_img_logit, center_label="Noise")
# fig.savefig("a.png")
# fig.show()

# 使用sea lion的背景噪音
# Img + TUAP
tuap = torch.unsqueeze(tuap_generator.uap, dim=0)

# 看看原始图片对应的标签式多少
outputs_imge1 = target_model(img1)
_, preds_img1 = torch.max(outputs_imge1, 1)


### Perpearing the inputs
img1_logit = target_model(img1).cpu().detach().numpy()
plot_img1 = img1[0].cpu().detach().numpy()
plot_img1 = np.transpose(plot_img1, (1, 2, 0))
plot_img1 = plot_img1 * std + mean


# 看看tuap对应的标签式多少
outputs_tuap = target_model((tuap-mean_tensor)/std_tensor)
_, preds_tuap = torch.max(outputs_tuap, 1)

# Original img + Other image
tuap_logit = target_model((tuap-mean_tensor)/std_tensor).cpu().detach().numpy()
plot_tuap = tuap[0].cpu().detach().numpy()
plot_tuap = np.transpose(plot_tuap, (1, 2, 0))
plot_tuap_normal = plot_tuap + 0.5

plot_tuap_amp = plot_tuap/2+0.5
tuap_range = np.max(plot_tuap_amp) - np.min(plot_tuap_amp)
plot_tuap_amp = plot_tuap_amp/tuap_range + 0.5
plot_tuap_amp -= np.min(plot_tuap_amp)

img_tuap = ((img1 * std_tensor + mean_tensor + tuap) - mean_tensor)/std_tensor
img_tuap_logit = target_model(img_tuap).cpu().detach().numpy()
tuap_logit_val = tuap_perturbed_net(img1)
plot_img_tuap = np.clip((plot_img1 + plot_tuap_normal - 0.5), 0, 1)

fig = plot(plot_img1, img1_logit, plot_tuap_amp, tuap_logit, plot_img_tuap, img_tuap_logit, center_label="targeted UAP")

fig.savefig("a.png")
# fig.show()

# 接下来要做的事情就是循环所有ImageNet中的验证数据，把每个数据作为背景图片与现有的img_tuap进行叠加融合，看看模型的预测结果，
# 我们预期结果应该是后门起主要作用，也就是说预期标签不会发生变化（我们认为这种做法应该是采用的蜕变测试进行操作，也需要进行深入的分析，
# 比如采用统计采样计算误报的概率上限是多少。）
# 再做实验室的时候考虑使用高斯噪声的影响，按照论文中提到的做法，高斯噪声应该会对结果产生影响，但是如果存在后门攻击的话，高斯噪声的作用为0，
# 是不是可以做一个数学上的证明，提升方法的复杂度，类似于差分隐私的做法呢
data_test_loader_val = torch.utils.data.DataLoader(data_test,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True)
# total_imgs=0
# right_imgs=0
# for step,(x,y) in enumerate(data_test_loader_val):
#     # 在含有tuap的图形基础上创建新的图形
#     img_tuap_newimg = ((x.cuda() * std_tensor + mean_tensor + img_tuap) - mean_tensor) / std_tensor
#
#     outputs = target_model(img_tuap_newimg)
#     _, preds = torch.max(outputs, 1)
#
#     total_imgs=total_imgs+1
#     if preds.item()== preds_tuap.item():
#         right_imgs=right_imgs+1
#     print("当前的图片为：{},使用的TUAP对应的标签为：{},初次使用的图片标签为：{}，预测值为：{},真实值为：{}，当前的准确率为：{}%".format(
#         total_imgs,preds_tuap.item(),preds_img1.item(), preds.item(), y.item(),right_imgs/total_imgs*100))

# 论文思路：这种遮蔽法可以解决tuap问题，却不能保障解决badnet问题；使用遮蔽以后的logit进行tsne展示，增加解释性与工作量；使用t检验看看两种分布是不是存在统计差别

# 看看不使用tuap的情况下，有什么样的结果
total_imgs=0
right_imgs=0
preds_newimg_list=[]
preds_tuap_newimg_list=[]
preds_noise_list=[]
# Img + Gaussian Noise Image
noise_img = torch.randn_like(img1) * 0.2
for step,(x,y) in enumerate(data_test_loader_val):
    # 在不含有tuap的图形基础上创建新的图形
    img_newimg = ((x.cuda() * std_tensor + mean_tensor + img1) - mean_tensor) / std_tensor
    outputs_newimg = target_model(img_newimg)
    _, preds_newimg = torch.max(outputs_newimg, 1)

    preds_newimg=preds_newimg.cpu().detach().numpy().tolist()
    preds_newimg_list.extend(preds_newimg)
    # print(preds_newimg_list)
    # 使用含有高斯噪音的图片
    img_noise = (( x.cuda()* std_tensor + mean_tensor +img1+ noise_img.cuda()) - mean_tensor) / std_tensor
    outputs_noise = target_model(img_noise)
    _, preds_noise = torch.max(outputs_noise, 1)
    preds_noise = preds_noise.cpu().detach().numpy().tolist()
    preds_noise_list.extend(preds_noise)

    # 在含有tuap的图形基础上创建新的图形
    img_tuap_newimg = ((x.cuda() * std_tensor + mean_tensor + img_tuap) - mean_tensor) / std_tensor
    outputs_tuap_newimg = target_model(img_tuap_newimg)
    _, preds_tuap_newimg = torch.max(outputs_tuap_newimg, 1)
    preds_tuap_newimg = preds_tuap_newimg.cpu().detach().numpy().tolist()
    preds_tuap_newimg_list.extend(preds_tuap_newimg)
    # print(preds_tuap_newimg_list)
    # exit()
    total_imgs=total_imgs+1
    # 测试期间，只选择其中的10000张图片即可
    if total_imgs % 200 ==0:
        print(total_imgs)
    if total_imgs==1000:
        break
    # print("当前的图片为：{},使用的TUAP对应的标签为：{},初次使用的图片标签为：{}，预测值为：{},真实值为：{}，当前的准确率为：{}%".format(
    #     total_imgs,preds_tuap.item(),preds_img1.item(), preds.item(), y.item(),right_imgs/total_imgs*100))

print(preds_newimg_list)
print(preds_tuap_newimg_list)
print(preds_noise_list)
from  scipy import  stats


print("无tuap的图片与有tuap的图片进行对比：")
print(stats.levene(preds_newimg_list, preds_tuap_newimg_list))					 # 进行levene检验
print(stats.ttest_ind(preds_newimg_list,preds_tuap_newimg_list,equal_var=False))

print("含有高斯噪音的图片与有tuap的图片进行对比：")
print(stats.levene(preds_noise_list, preds_tuap_newimg_list))					 # 进行levene检验
print(stats.ttest_ind(preds_noise_list,preds_tuap_newimg_list,equal_var=False))


