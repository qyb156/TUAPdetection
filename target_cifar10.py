from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from tests.utils import (
    TestBase,
    get_image_classifier_pt,
    get_cifar10_image_classifier_pt,
)
import pandas as pd
import datetime

from art.attacks.evasion import TargetedUniversalPerturbation
logger = logging.getLogger(__name__)

from torchvision import datasets,transforms
from torch.utils.data import DataLoader
train_batch_size=1000
test_batch_size=100

MR_relations=["noise","random"]

# 导入数据
train_dataset = datasets.CIFAR10(root = './data/', train = True,
                               transform = transforms.ToTensor(), download = True)
test_dataset = datasets.CIFAR10(root = './data/', train = False,
                               transform = transforms.ToTensor(), download = True)
# 加载数据，打乱顺序并一次给模型100个数据
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
images_train, labels_train = next(iter(train_loader))




# x_train_mnist = np.swapaxes(x_train_mnist, 1, 3).astype(np.float32)
# x_test_mnist = np.swapaxes(x_test_mnist, 1, 3).astype(np.float32)

x_train_mnist = images_train.cpu().numpy()

# print("y_test_mnist 原始测试数据集的结果:",y_test_mnist)
# x_test_original = x_test_mnist.copy()

from torchvision import models
# Build PyTorchClassifier
ptc = get_cifar10_image_classifier_pt()

# set target label
target = 9
print("目标是：",target)
y_target = np.zeros([len(x_train_mnist), 10])
for i in range(len(x_train_mnist)):
    y_target[i, target] = 1.0

# Attack
up = TargetedUniversalPerturbation(
    ptc, max_iter=1, attacker="fgsm", attacker_params={"eps": 0.3, "targeted": True, "verbose": False}
)
x_train_mnist_adv = up.generate(x_train_mnist, y=y_target)
print("在训练数据集上的攻击成功率为：",up.fooling_rate)
# self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)

# 最终的结果
final_resluts_pd=pd.DataFrame([])

zeta_list=[0.01,0.1,1]
verified_imgs_count=100

for zetaindex in range(len(zeta_list)):
    # 设定超参数
    zeta=zeta_list[zetaindex]
    detection_success=0

    tuap_list=[]
    non_tuap_list = []
    for t in range(1, verified_imgs_count + 1):
        print(zeta,t)
        starttime=datetime.datetime.now()
        # 注意这里是每次取出一个图片。
        images_test, labels_test = next(iter(test_loader))
        x_test_mnist = images_test.cpu().numpy()
        y_test_mnist = labels_test.cpu().numpy()
        # 合成一个有后门的图像
        x_test_mnist_adv = x_test_mnist + up.noise
        # print("up.noise.shape",up.noise.shape)
        # print("up.noise",up.noise)
        #
        # image=up.noise
        # image=np.squeeze(image,axis=0)
        # # print(image.shape)
        # image = np.transpose(image, (1,2,0))
        # # print(image.shape)
        # import  matplotlib.pyplot as plt
        # from  PIL import  Image
        # image=Image.fromarray(np.uint8(image))

        train_y_pred = np.argmax(ptc.predict(x_train_mnist_adv), axis=1)
        test_y_pred = np.argmax(ptc.predict(x_test_mnist_adv), axis=1)
        # print("对有后门图像添加扰动的预测结果，test_y_pred:",test_y_pred)
        # assertFalse((np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all())
        # print("(np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all()",(y_test_mnist == test_y_pred).all())
        # self.assertFalse((np.argmax(self.y_train_mnist, axis=1) == train_y_pred).all())

        # 根据画图的情况来看，500个图片即可体现出来效果。
        count_shumu = 100
        # 先初始化统计结果
        total_tuap = 0
        total_nontuap = 0
        # 扰动数据
        test_loader_random = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
        # 先初始化统计结果
        total_tuap = 0
        total_nontuap = 0

        for i in range(1, count_shumu + 1):
            # images_train, labels_train = next(iter(train_loader))
            images_random, labels_test2 = next(iter(test_loader_random))
            x_test_mnist_random = images_random.cpu().numpy()
            y_test_mnist2 = labels_test2.cpu().numpy()

            x_test_mnist_adv2 = x_test_mnist_adv + zeta * x_test_mnist_random
            test_y_pred2 = np.argmax(ptc.predict(x_test_mnist_adv2), axis=1)
            # print("对有后门图像添加二次随机扰动以后的结果:",test_y_pred2)
            # res=np.sum( np.equal(test_y_pred2,test_y_pred).astype(int))
            # print("对有后门图像添加二次随机扰动以后的结果:",res/test_batch_size*100)
            # 测试正常图片添加扰动以后的情况
            x_test_mnist_normal = x_test_mnist + zeta * x_test_mnist_random
            test_y_pred_normal = ptc.predict(x_test_mnist_normal)
            test_y_pred3 = np.argmax(test_y_pred_normal, axis=1)
            # print("对正常图片添加随机扰动以后的结果:",test_y_pred3)
            # print(np.equal(test_y_pred3,y_test_mnist).astype(int))
            # res=np.sum( np.equal(test_y_pred3,y_test_mnist).astype(int))
            # print("对正常图片添加随机扰动以后的结果:",res/test_batch_size*100)
            if test_y_pred2.item()==target:
                total_tuap+=1
            # 这里的逻辑是，应该统计的是出现次数最多的图像。我们知道目标以后，就可以简化为，统计干净图像的数量。因为即便是随机的，我们
            # 认为干净图像的数量仍是出现最多次数的，可以表示最大类别的图像的数量。
            if test_y_pred3.item()==labels_test.item():
                total_nontuap+=1

        if total_nontuap<=total_tuap:
            detection_success+=1
        tuap_list.append(total_tuap/count_shumu)
        non_tuap_list.append(total_nontuap / count_shumu)
        endtime=datetime.datetime.now()
        final_resluts_pd = final_resluts_pd.append([[zeta_list[zetaindex], "random", np.mean(non_tuap_list),
                                                     np.var(non_tuap_list), np.mean(tuap_list), np.var(tuap_list),
                                                     str((endtime - starttime).seconds / verified_imgs_count),
                                                     detection_success / verified_imgs_count]])
        final_resluts_pd.to_csv("results/" + "final_cifar10.csv")