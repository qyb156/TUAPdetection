from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import sklearn
from sklearn.decomposition import PCA

torch.autograd.set_detect_anomaly(True)

# Training settings
parser = argparse.ArgumentParser(description='generative adversarial perturbations')
parser.add_argument('--imagenetTrain', type=str, default='imagenet/train', help='ImageNet train root')
parser.add_argument('--imagenetVal', type=str, default='imagenet/val', help='ImageNet val root')
parser.add_argument('--batchSize', type=int, default=30, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: "adam" or "sgd"')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--MaxIter', type=int, default=50, help='Iterations in each Epoch')
parser.add_argument('--MaxIterTest', type=int, default=10000, help='Iterations in each Epoch')
parser.add_argument('--mag_in', type=float, default=10.0, help='l_inf magnitude of perturbation')
parser.add_argument('--expname', type=str, default='test_incv3_universal_targeted_linf10_twogpu', help='experiment name, output folder')
parser.add_argument('--checkpoint', type=str, default='', help='path to starting checkpoint')
parser.add_argument('--foolmodel', type=str, default='incv3', help='model to fool: "incv3", "vgg16", or "vgg19"')
parser.add_argument('--mode', type=str, default='test', help='mode: "train" or "test"')
parser.add_argument('--perturbation_type', type=str, default="universal", help='"universal" or "imdep" (image dependent)')
parser.add_argument('--target', type=int, default=805, help='target class: -1 if untargeted, 0..999 if targeted')
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
parser.add_argument('--path_to_U_noise', type=str, default='', help='path to U_input_noise.txt (only needed for universal)')
parser.add_argument('--explicit_U', type=str, default='test_incv3_universal_targeted_linf10_twogpu/U_out/U_epoch_10_top1target_48.64130401611328.pth', help='Path to a universal perturbation to use')
opt = parser.parse_args()

import  pickle

print(opt)

# train loss history
train_loss_history = []
test_loss_history = []
test_acc_history = []
test_fooling_history = []
best_fooling = 0
itr_accum = 0

MaxIter = opt.MaxIter
MaxIterTest = opt.MaxIterTest

# define normalization means and stddevs
model_dimension = 299 if opt.foolmodel == 'incv3' else 256
center_crop = 299 if opt.foolmodel == 'incv3' else 224

mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean_arr,
                                 std=stddev_arr)

data_transform = transforms.Compose([
    transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor(),
    normalize,
])

print('===> Loading datasets')

# if opt.mode == 'train':
train_set = torchvision.datasets.ImageFolder(root = opt.imagenetTrain, transform = data_transform)
# 这里出错了，改为0
# training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=1, shuffle=True)

# magnitude
mag_in = opt.mag_in

def TrainBaseline():
    list_x=[]
    list_y = []
    if opt.explicit_U:
        U_loaded = torch.load(opt.explicit_U)
        U_loaded = U_loaded.expand(opt.testBatchSize, U_loaded.size(1), U_loaded.size(2), U_loaded.size(3))
        delta_im = normalize_and_scale(U_loaded, 'test')
    # 注意是从训练数据集中选择数据，不能使用验证数据集，以免数据泄露
    for itr, (image, label) in enumerate(training_data_loader):
        itr=itr+1
        # 训练2000个数据
        if itr > 3000:
            break
        # 既是模仿论文中数据的分布，也是模仿现实的情况，。实际情况中，有后门的数据少之又少
        if itr%600==0:
            # 这是合成了后门的图像，设定标签为-1
            recons = torch.add(image.cuda(), delta_im[0:image.size(0)].cuda())
            # do clamping per channel
            for cii in range(3):
                recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(image[:,cii,:,:].min(), image[:,cii,:,:].max())
            recons = recons.cuda().detach().cpu().numpy().flatten().tolist()
            list_x.append(recons)
            list_y.append(0)
        else:
            # 这是原有的图像，设定标签为1
            image = image.cuda().cpu().numpy().flatten().tolist()
            list_x.append(image)
            list_y.append(1)

    pca=PCA(n_components=200)
    list_x=pca.fit_transform(np.array(list_x))

    maxitr=2000
    print("PCA转换完成，开始训练模型,maxitr=",maxitr)
    svc = sklearn.svm.LinearSVC (max_iter=maxitr)
    # svc=sklearn.svm.SVC(max_iter=maxitr)
    svc.fit(X= list_x,y=list_y)
    with open("model/svc.pickle",'wb') as f:
        pickle.dump(svc,f)
    with open("model/pca.pickle", 'wb') as f:
        pickle.dump(pca, f)
    print("线性支持向量机训练结束了。")

    with open("model/svc.pickle",'rb') as f:
        clf2= pickle.load(f)
    with open("model/pca.pickle",'rb') as f:
        pca= pickle.load(f)
    print("模型读取成功了。")


def normalize_and_scale(delta_im, mode='train'):
    if opt.foolmodel == 'incv3':
        delta_im = nn.ConstantPad2d((0,-1,-1,0),0)(delta_im) # crop slightly to match inception

    delta_im = delta_im + 1 # now 0..2
    delta_im = delta_im * 0.5 # now 0..1

    # normalize image color channels
    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    bs = opt.batchSize if (mode == 'train') else opt.testBatchSize
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = mag_in/(255.0*stddev_arr[ci])
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im

print('开始初始化基线模型...')
TrainBaseline()
