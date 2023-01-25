from __future__ import print_function
import argparse
import os
from math import log10
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')
import torchvision
import torch
print(torch.cuda.is_available())
exit()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from models import ResnetGenerator, weights_init
from material.models.generators import ResnetGenerator, weights_init
from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
import math
import torchvision.transforms as transforms
import numpy as np
from sklearn.decomposition import PCA
import  pickle
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

print(opt)

#if not torch.cuda.is_available():
#    raise Exception("No GPU found.")

# train loss history
train_loss_history = []
test_loss_history = []
test_acc_history = []
test_fooling_history = []
best_fooling = 0
itr_accum = 0

# make directories
if not os.path.exists(opt.expname):
    os.mkdir(opt.expname)

if opt.perturbation_type == 'universal':
    if not os.path.exists(opt.expname + '/U_out'):
        os.mkdir(opt.expname + '/U_out')

cudnn.benchmark = True
torch.cuda.manual_seed(opt.seed)

MaxIter = opt.MaxIter
MaxIterTest = 1000
gpulist = [int(i) for i in opt.gpu_ids.split(',')]
n_gpu = len(gpulist)
print('Running with n_gpu: ', n_gpu)

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

if opt.mode == 'train':
    train_set = torchvision.datasets.ImageFolder(root = opt.imagenetTrain, transform = data_transform)
    # 这里出错了，改为0
    # training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True)

test_set = torchvision.datasets.ImageFolder(root = opt.imagenetVal, transform = data_transform)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.testBatchSize, shuffle=True)

if opt.foolmodel == 'incv3':
    pretrained_clf = torchvision.models.inception_v3(pretrained=True)
elif opt.foolmodel == 'vgg16':
    pretrained_clf = torchvision.models.vgg16(pretrained=True)
elif opt.foolmodel == 'vgg19':
    pretrained_clf = torchvision.models.vgg19(pretrained=True)

pretrained_clf = pretrained_clf.cuda(gpulist[0])

pretrained_clf.eval()
pretrained_clf.volatile = True

# magnitude
mag_in = opt.mag_in

print('===> Building model')

if not opt.explicit_U:
    # will use model paralellism if more than one gpu specified
    netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)

    # resume from checkpoint if specified
    if opt.checkpoint:
        if os.path.isfile(opt.checkpoint):
            print("=> loading checkpoint '{}'".format(opt.checkpoint))
            netG.load_state_dict(torch.load(opt.checkpoint, map_location=lambda storage, loc: storage))
            print("=> loaded checkpoint '{}'".format(opt.checkpoint))
        else:
            print("=> no checkpoint found at '{}'".format(opt.checkpoint))
            netG.apply(weights_init)
    else:
        netG.apply(weights_init)

    # setup optimizer
    if opt.optimizer == 'adam':
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.optimizer == 'sgd':
        optimizerG = optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9)

    criterion_pre = nn.CrossEntropyLoss()
    criterion_pre = criterion_pre.cuda(gpulist[0])

    # fixed noise for universal perturbation
    if opt.perturbation_type == 'universal':
        noise_data = np.random.uniform(0, 255, center_crop * center_crop * 3)
        if opt.checkpoint:
            if opt.path_to_U_noise:
                noise_data = np.loadtxt(opt.path_to_U_noise)
                np.savetxt(opt.expname + '/U_input_noise.txt', noise_data)
            else:
                noise_data = np.loadtxt(opt.expname + '/U_input_noise.txt')
        else:
            np.savetxt(opt.expname + '/U_input_noise.txt', noise_data)
        im_noise = np.reshape(noise_data, (3, center_crop, center_crop))
        im_noise = im_noise[np.newaxis, :, :, :]
        im_noise_tr = np.tile(im_noise, (opt.batchSize, 1, 1, 1))
        noise_tr = torch.from_numpy(im_noise_tr).type(torch.FloatTensor).cuda(gpulist[0])

        im_noise_te = np.tile(im_noise, (opt.testBatchSize, 1, 1, 1))
        noise_te = torch.from_numpy(im_noise_te).type(torch.FloatTensor).cuda(gpulist[0])

def detect(delta_im,image_ori,class_label_ori,batch_size2=50,MaxIterTest2=2):
    # 这是我们自己的代码，从验证集中再取出100张随机图像，分别与原始图像，合成图像再次合成，计算得到的结果序列
    testing_data_loader2 = DataLoader(dataset=test_set, num_workers=0, batch_size=batch_size2, shuffle=True)

    if not opt.explicit_U:
        netG.eval()
    correct_recon = 0
    correct_orig = 0
    total = 0
    ratio=0.01

    for itr, (image, class_label) in enumerate(testing_data_loader2):
        if itr > MaxIterTest2:
            break

        image = image.cuda(gpulist[0])
        recons = torch.add(ratio*image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))
        # print(recons.size())
        # exit()
        # 在当前的图像上使用原有的clean图像进行扰动
        image_from_ori = torch.add( ratio * image.cuda(gpulist[0]),image_ori[0:image.size(0)].cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(image[:,cii,:,:].min(), image[:,cii,:,:].max())
            image_from_ori[:, cii, :, :] = image_from_ori[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        outputs_recon = pretrained_clf(recons.cuda(gpulist[0]))
        _, predicted_recon = torch.max(outputs_recon, 1)
        # 对生成的图像进行预测
        outputs_from_orig = pretrained_clf(image_from_ori.cuda(gpulist[0]))
        _, predicted_from_orig = torch.max(outputs_from_orig, 1)
        total += image.size(0)
        correct_recon += (predicted_recon.cpu().numpy() == opt.target).sum()
        correct_orig += (predicted_from_orig.cpu().numpy() == class_label_ori).sum()

    print('在后门图像上二次扰动以后成功判别的几率: %.2f%%' % (100.0 * float(correct_recon) / float(total)))
    print('在干净图像上二次扰动以后成功判别的几率: %.2f%%' % (100.0 * float(correct_orig) / float(total)))
    if correct_orig <= correct_recon:
        return [True, 100.0 * float(correct_recon) / float(total), 100.0 * float(correct_orig) / float(total)]
    else:
        return [False, 0, 0]

def test():
    success=0
    attack_success=0
    baseline_success = 0

    success_list = []
    attack_success_list = []
    baseline_success_list = []
    # 用于记录lsr
    backdoor_list = []
    clean_list = []

    if not opt.explicit_U:
        netG.eval()

    if opt.perturbation_type == 'universal':
        if opt.explicit_U:
            U_loaded = torch.load(opt.explicit_U)
            U_loaded = U_loaded.expand(opt.testBatchSize, U_loaded.size(1), U_loaded.size(2), U_loaded.size(3))
            delta_im = normalize_and_scale(U_loaded, 'test')
            # print("从训练数据中加载了模型，是不是生产的扰动呢？")
        else:
            delta_im = netG(noise_te)
            delta_im = normalize_and_scale(delta_im, 'test')

    if os.path.exists("model/svc.pickle") and os.path.exists("model/pca.pickle"):
        with open("model/svc.pickle", 'rb') as f:
            svc = pickle.load(f)

        with open("model/pca.pickle", 'rb') as f:
            pca = pickle.load(f)
        print("SVC与PCA模型读取成功了。")
    else:
        print("SVC或者PCA 模型未加载成功，需要首先训练该模型。")
        exit()

    for itr, (image, class_label) in enumerate(testing_data_loader):


        itr=itr+1
        if itr > MaxIterTest:
            break

        image = image.cuda(gpulist[0])
        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(image[:,cii,:,:].min(), image[:,cii,:,:].max())

        # 看看当前图像的是否能够攻击成功
        outputs_recon = pretrained_clf(recons.cuda(gpulist[0]))
        _, predicted_recon = torch.max(outputs_recon, 1)
        if predicted_recon.cpu().numpy().item() == opt.target:
            # print(predicted_recon.cpu().numpy().item())
            attack_success+=1

        # 使用支持向量机对合成的有后门图像进行预测，预测的结果应该是-1，表示是有后门的
        recons2 = recons.cuda().detach().cpu().numpy().flatten().tolist()
        recons2 = pca.transform(np.array([recons2]))
        predicted_y=svc.predict(recons2)
        print("SVM预测对抗样本的结果：",predicted_y)
        if predicted_y[0] == 0:
            baseline_success+=1

        image2 = image.cuda().detach().cpu().numpy().flatten().tolist()
        image2 = pca.transform(np.array([image2]))
        predicted_y2 = svc.predict(image2)
        print("SVM预测正常样本的结果：", predicted_y2)

        start = datetime.datetime.now()
        # 对每张合成后的图片进行验证。
        print("对第{}张图片进行检验：".format(itr))
        result = detect(recons, image, class_label.item())
        end = datetime.datetime.now()
        all_seconds = (end - start).seconds
        print("一次扰动时间为：", all_seconds )
        res = result[0]
        backdoor = result[1]
        clean = result[2]
        if res == True:
            success += 1
            backdoor_list.append(backdoor)
            clean_list.append(clean)

        print("当前图像被攻击的成功率：", attack_success / itr * 100)
        print("SVM基线探测器预测的成功率：", baseline_success / itr * 100)
        print("我们探测算法的当前成功率：", success / itr * 100)

        attack_success_list.append(attack_success / itr * 100)
        baseline_success_list.append(baseline_success / itr * 100)
        success_list.append(success / itr * 100)

        if itr%20==0:
            print(attack_success_list)
            print(baseline_success_list)
            print(success_list)

        pd.DataFrame(attack_success_list).to_csv("attack_success.csv")
        pd.DataFrame(baseline_success_list).to_csv("baseline_success.csv")
        pd.DataFrame(success_list).to_csv("success.csv")

        # 导出lsr
        pd.DataFrame(backdoor_list).to_csv("backdoor_list_random.csv")
        pd.DataFrame(clean_list).to_csv("clean_list_random.csv")

        print("distance of LSR:", np.array(backdoor_list).sum()/np.array(clean_list).sum())



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
            gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im


def checkpoint_dict(epoch):
    netG.eval()
    global best_fooling
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    task_label = "foolrat" if opt.target == -1 else "top1target"

    net_g_model_out_path = opt.expname + "/netG_model_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(test_fooling_history[epoch-1])
    if opt.perturbation_type == 'universal':
        u_out_path = opt.expname + "/U_out/U_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(test_fooling_history[epoch-1])
    if test_fooling_history[epoch-1] > best_fooling:
        best_fooling = test_fooling_history[epoch-1]
        torch.save(netG.state_dict(), net_g_model_out_path)
        if opt.perturbation_type == 'universal':
            torch.save(netG(noise_te[0:1]), u_out_path)
        print("Checkpoint saved to {}".format(net_g_model_out_path))
    else:
        print("No improvement:", test_fooling_history[epoch-1], "Best:", best_fooling)


def print_history():
    # plot history for training loss
    if opt.mode == 'train':
        plt.plot(train_loss_history)
        plt.title('Model Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.legend(['Training Loss'], loc='upper right')
        plt.savefig(opt.expname+'/reconstructed_loss_'+opt.mode+'.png')
        plt.clf()

    # plot history for classification testing accuracy and fooling ratio
    plt.plot(test_acc_history)
    plt.title('Model Testing Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Testing Classification Accuracy'], loc='upper right')
    plt.savefig(opt.expname+'/reconstructed_acc_'+opt.mode+'.png')
    plt.clf()

    plt.plot(test_fooling_history)
    plt.title('Model Testing Fooling Ratio')
    plt.ylabel('Fooling Ratio')
    plt.xlabel('Epoch')
    plt.legend(['Testing Fooling Ratio'], loc='upper right')
    plt.savefig(opt.expname+'/reconstructed_foolrat_'+opt.mode+'.png')
    print("Saved plots.")

# if opt.mode == 'train':
#     for epoch in range(1, opt.nEpochs + 1):
#         train(epoch)
#         print('Testing....')
#         test()
#         checkpoint_dict(epoch)
#     print_history()
# elif opt.mode == 'test':
print('开始探测是否存在后门...')
import datetime
start=datetime.datetime.now()
test()
end=datetime.datetime.now()
all_seconds=(end-start).seconds
print("高斯扰动平均时间为：",all_seconds/MaxIterTest)
