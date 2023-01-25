# %matplotlib inline
from  __future__ import  division
import os
import numpy as np
import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)
# exit()

from matplotlib import pyplot as plt
from collections import OrderedDict
import torch.nn as nn
from networks.uap import UAP
from utils.data import get_data_specs, get_data
from utils.utils import softmax, get_imagenet_dicts
from utils.network import get_network
import pandas as pd
# from scipy import  stats


MODEL_ARCH = "vgg19"
TARGET_CLASS = 150
TUAP_RESULT_PATH = "./vgg19_sea_lion"

datasets=["imagenet"]
import datetime

NGPU=1
USE_CUDA=True
pcc1_list=[]
pcc2_list=[]
MR_relations=["noise","random"]
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

    return fig,pcc1[0, 1],pcc2[0, 1]
# 最终的结果
final_resluts_pd=pd.DataFrame([])

zeta_list=[0.01,0.1,1]

for zetaindex in range(len(zeta_list)):
    # 设定超参数
    zeta=zeta_list[zetaindex]
    # 遍历蜕变关系
    for m in range(len(MR_relations)):
        detection_success = 0
        starttime = datetime.datetime.now()
        # 测试1000张随机图片，检测一下分类的结果，另外还要考虑两种蜕变关系，多个数据集。
        verified_imgs_count=40
        # 定义两个预测结果的列表，含有tuap与不含有tuap的结果
        tuap_list = []
        non_tuap_list = []
        for t in range(1, verified_imgs_count + 1):

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

            img1, lbl1 = next(test_iter)
            if USE_CUDA:
                img1 = img1.cuda()
                lbl1 = lbl1.cuda()

            # 根据画图的情况来看，500个图片即可体现出来效果。
            count_shumu=100
            # 先初始化统计结果
            total_tuap = 0
            total_nontuap = 0
            for i in range(1,count_shumu+1):
                # 使用sea lion的背景噪音
                tuap = torch.unsqueeze(tuap_generator.uap, dim=0)
                # tuap
                tuap_logit = target_model((tuap - mean_tensor) / std_tensor).cpu().detach().numpy()
                tuap_label = torch.argmax(target_model((tuap - mean_tensor) / std_tensor))
                # 在随机图像上叠加tuap
                img_tuap = ((img1 * std_tensor + mean_tensor + tuap) - mean_tensor) / std_tensor

                if MR_relations[m]=="random":
                    # 我们的工作是在原有的图片的基础上再叠加一层，这是进行蜕变关系分析的基础。
                    # 首先再随机选择一个其他的图片
                    img_random, lbl_random = next(test_iter)
                    if USE_CUDA:
                        img_random = img_random.cuda()
                        lbl_random = lbl_random.cuda()
                    img_random =(( zeta*img_random * std_tensor + mean_tensor +img1 ) - mean_tensor) / std_tensor
                    img_MR_label = torch.argmax(target_model(img_random))

                    img_tuap_random = (( zeta*img_random * std_tensor + mean_tensor + img_tuap ) - mean_tensor)/std_tensor
                    img_tuap_MR_label = torch.argmax(target_model(img_tuap_random))

                    if img_tuap_MR_label.item()==img_MR_label.item():
                        total_nontuap=total_nontuap+1
                    if tuap_label.item() == img_tuap_MR_label.item():
                        total_tuap = total_tuap + 1
                else:
                    # 在原始图片上加上高斯噪音
                    noise_img = torch.randn_like(img1)
                    img_noise_img = (( noise_img * std_tensor + mean_tensor + img1) - mean_tensor)/std_tensor
                    img_noise_img_logit = target_model(img_noise_img).cpu().detach().numpy()
                    img_MR_label = torch.argmax(target_model(img_noise_img))

                    img_tuap_random = ((zeta * noise_img * std_tensor + mean_tensor + img_tuap) - mean_tensor) / std_tensor
                    img_tuap_MR_label = torch.argmax(target_model(img_tuap_random))
                    # 代码重复为的是后面好阅读代码。
                    if img_tuap_MR_label.item() == img_MR_label.item():
                        total_nontuap = total_nontuap + 1
                    if tuap_label.item() == img_tuap_MR_label.item():
                        total_tuap = total_tuap + 1
                # print("total_nontuap:" + str(total_nontuap))
                # print("total_tuap:"+str(total_tuap))
            # 这里的逻辑关系是：count_shumu张随机图片在待验证图片上面进行扰动，记录扰动的结果
            non_tuap_list.append(total_nontuap // count_shumu)

            tuap_list.append(total_tuap // count_shumu)
            # print("total_tuap,{},count_shumu{},total_tuap//count_shumu:{}".format(total_tuap,count_shumu,total_tuap//count_shumu))
            # print(tuap_list)
            pd.DataFrame(tuap_list).to_csv("results/"+"tuap_rq1_"+datasets[0]+"_"+MR_relations[m]+".csv",index=False)
            pd.DataFrame(non_tuap_list).to_csv("results/"+"non_tuap_rq1_"+datasets[0]+"_"+MR_relations[m]+".csv",index=False)

            if total_tuap>=total_nontuap:
                detection_success+=1
        print("超参数为：{},蜕变关系为:{},待验证图片为：{}张,探测的成功率为：{}。".format(zeta_list[zetaindex],MR_relations[m],verified_imgs_count,detection_success/verified_imgs_count))
            # print("第{}张图片处理完成，数据集是:{},蜕变关系是：{}".format(t,datasets[0],MR_relations[m]))
        # 存储最终的统计结果
        # 这里的逻辑关系是，针对每种蜕变关系，记录含有tuap和不含有tuap的结果。
        endtime = datetime.datetime.now()
        final_resluts_pd=final_resluts_pd.append([[zeta_list[zetaindex],MR_relations[m],np.mean(non_tuap_list),
                                                   np.var(non_tuap_list),np.mean(tuap_list),np.var(tuap_list),
                                                   str((endtime - starttime).seconds/verified_imgs_count),
                                                   detection_success/verified_imgs_count]])

        print(MR_relations[m]+",平均处理一张图片的时间为："+str((endtime - starttime).seconds/verified_imgs_count))
    final_resluts_pd.to_csv("results/"+"final_imagenet.csv")

