import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' ############################
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init

from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,DiceLoss,JaccardLoss,mae     ###################
from cal import calculate_metrics
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import timeit
from arch.ae import kiunet,kinetwithsk,unet,autoencoder, reskiunet,densekiunet, kiunet3d,kitenet
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

#参数
parser = argparse.ArgumentParser(description='KiU-Net')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH',
                    help='path to root path of lfw dataset (default: ../lfw)')
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int,default = 5)

parser.add_argument('--modelname', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str, ############################
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--save', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--model', default='kiunet', type=str,
                    help='model name')
parser.add_argument('--direc', default='./brainus_OC_udenet', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
#parser.add_argument('--edgeloss', default='j_loss', type=str)#######################

args = parser.parse_args()

aug = args.aug
direc = args.direc
modelname = args.modelname
#losstype= args.edgeloss

#初始化权重
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)

#random noise
def add_noise(img):
    noise = torch.randn(img.size()) * 0.1
    noisy_img = img + noise.cuda()
    return noisy_img
     

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)

train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda")

if modelname == "unet":
    model = unet()
elif modelname =="autoencoder":
    model =autoencoder()
elif modelname == "kiunet":
    model = kiunet()
elif modelname == "kinetwithsk":
    model = kinetwithsk()
elif modelname == "kitenet":
    model = kitenet()
elif modelname == "reskiunet":
    model = reskiunet()
elif modelname == "densekiunet":
    model = densekiunet()
elif modelname == "kiunet3d":
    model = kiunder3d()
elif modelname == "pspnet":
    model = psp.PSPNet(layers=5, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=False).cuda()

#如果有多个GPU可用，则使用DataParallel
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)
model.apply(weight_init)#############################
#print(model)
bestdice=0
#损失函数，优化器，指标
criterion = LogNLLLoss()
dice_loss_func = DiceLoss()
jaccard_loss_func = JaccardLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)
# 创建学习率调度器
#scheduler = StepLR(optimizer, step_size=50, gamma=0.9)  # 每10个epoch将学习率乘以0.1##########################
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))
#model.train()
print("start training")


losses = []

for epoch in range(args.epochs):

    epoch_running_loss = 0
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):        
        
        ###augmentations

        X_batch = Variable(X_batch.to(device ='cuda'))#(1,3,128,128),输入彩色的切片

        y_batch = Variable(y_batch.to(device='cuda'))#（1，128，128）batch_size,x,y）
        #data_type = y_batch.dtype
        #数据增强
        #numr = randint(0,9)
        # numr = randint(2, 4)
        # if aug=='on':
        # #执行水平和垂直翻转
        #     if numr == 2:
        #         # print(X_batch,y_batch)
        #         X_batch = torch.flip(X_batch,[2,3])
        #         y_batch = torch.flip(y_batch,[1,2])
        #         # print(X_batch,y_batch)
        # #垂直，水平翻转
        #     elif numr ==3:
        #         X_batch = torch.flip(X_batch,[3,2])
        #         y_batch = torch.flip(y_batch,[2,1])
        # #加噪声
        #     elif numr==4:
        #         X_batch = add_noise(X_batch)
        #         # y_batch = add_noise(y_batch)

        # noisy_in = add_noise(X_batch)
        # ===================forward==============================
        output = model(X_batch) #tensor #(batch_size,classes,x,y)
        #output1=output[:, 0, :, :]
        # tmp = output.detach().cpu().numpy()
        # tmp2 = y_batch.detach().cpu().numpy()
        #y1=tmp[0, 0, :, :] #前景
        #y2=tmp[0, 1, :, :] #背景

        # # ===============================================================
        # tmp=tmp[0, 1, :, :]
        # tmp[tmp>=0.5] = 1 #前景
        # tmp[tmp<0.5] = 0
        # tmp = tmp.astype(int)
        #
        #
        # tmp2=tmp2[0, :, :]
        # tmp2[tmp2>0] = 1
        # tmp2[tmp2<=0] = 0
        # tmp2 = tmp2.astype(int)
        #
        # #print(np.unique(tmp2))
        # yHaT = tmp #预测 (128,128)
        # yval = tmp2 #标签

        # 保存图片
        # yval[yval == 1] = 255
        # yHaT[yHaT == 1] = 255
        # cv2.imwrite('C:/Users/27612/Desktop/KiU-Net-pytorch-master/KiU-Net-pytorch-master/test'+'/gt.png', yval[0,:,:])
        # cv2.imwrite('C:/Users/27612/Desktop/KiU-Net-pytorch-master/KiU-Net-pytorch-master/test'+'/pre.png', yHaT[0, 1, :, :])
    # # 计算预测结果和真实标签之间的边缘损失
    #     if losstype =='dice_loss':##F1
    #         # dice loss
    #         d_loss = Dice_Loss()
    #         edgeloss = d_loss.dice_loss(yval, yHaT)
    #     elif losstype =='jaccard_loss':#iou
    #        # jaccard loss
    #         j_loss = Jaccard_Loss()
    #         edgeloss = j_loss.jaccard_loss(yval, yHaT)
    #     elif losstype =='mae_loss':
    #         edgeloss = mae(yHaT,yval)
    #
    #     else:
    #         edgeloss = 0

        #========================loss==================================

        ce_loss = criterion(output, y_batch)
        dice_loss = dice_loss_func.dice_loss(output, y_batch)
        jaccard_loss = jaccard_loss_func.jaccard_loss(output, y_batch)
        #损失加权
        loss =criterion(output, y_batch)
        # ===================backward====================
        optimizer.zero_grad()#清零
        loss.backward()#误差反向传播
        optimizer.step()#参数更新

        epoch_running_loss += loss.item()#取高精度的值
        #scheduler.step()  # 更新学习率

    # ===================输出指标========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, args.epochs, epoch_running_loss/(batch_idx+1)))
    # print("Fsc:", f1_train / (batch_idx+1))
    # print("MIU:", miou_train / (batch_idx+1))
    # print("PA:", pa_train / (batch_idx+1))
    #=======================输出loss图=============================##################
    epoch_loss=epoch_running_loss / (batch_idx + 1)
    losses.append(epoch_loss)



# ===========================validation=======================================

    if (epoch % args.save_freq) ==0 or (epoch == args.epochs - 1):###################
        count=0
        f1 = 0
        miou = 0
        pa = 0
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            #print(batch_idx)
            if isinstance(rest[0][0], str):         #保存图片
                        image_filename = rest[0][0]
            else:
                        image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch = Variable(X_batch.to(device='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            # start = timeit.default_timer()
            y_out = model(X_batch)
            # y_out1=y_out[0, 0, :, :]#前景
            # y_out2=y_out[0, 1, :, :]#背景
            # stop = timeit.default_timer()
            # print('Time: ', stop - start)#查看一批次的训练时间

            tmp = y_out.detach().cpu().numpy()
            tmp = tmp[0, 1, :, :]
            tmp[tmp >= 0.5] = 1
            tmp[tmp < 0.5] = 0
            tmp = tmp.astype(int)

            tmp2 = y_batch.detach().cpu().numpy()
            tmp2 = tmp2[0, :, :]
            tmp2[tmp2 > 0] = 1
            tmp2[tmp2 <= 0] = 0
            tmp2 = tmp2.astype(int)

            # print(np.unique(tmp2))
            yHaT = tmp #(1,2,128,128?)
            yval = tmp2

            epsilon = 1e-20

            del X_batch, y_batch,tmp,tmp2, y_out

            count = count + 1
            #print("count:",count)
            yHaT[yHaT==1] =255 #保存图片
            yval[yval==1] =255 #保存图片
            # yHaT = torch.tensor(yHaT)
            # yHaT=yHaT.to(torch.int64)
            # yval = torch.tensor(yval)
            # yval = yval.to(torch.int64)
            # 计算指标


            tmiou, tpa, tf1 = calculate_metrics(yHaT, yval)########################

            # 累积指标
            f1 += tf1
            miou += tmiou
            pa += tpa


            ##保存正在训练的模型对测试集进行预测的结果
            fulldir = direc+"/{}/".format(epoch)#保存图片
            # print(fulldir+image_filename)
            if not os.path.isdir(fulldir):#保存图片

                os.makedirs(fulldir)

            cv2.imwrite(fulldir+image_filename, yHaT)#保存预测结果
            # cv2.imwrite('C:/Users/27612/Desktop/KiU-Net-pytorch-master/KiU-Net-pytorch-master/test'+'/gt_{}.png'.format(count), yval)
        fulldir = direc+"/{}/".format(epoch)
        torch.save(model.state_dict(), fulldir+args.model+".pth")

        print("F1:", f1 / count)
        print("mIou:", miou / count)
        #print("PA:", pa / count)

        F1=f1 / count
        if bestdice<F1:
            bestdice = F1
            print('epoch:', epoch)
            print("bestdice = {}".format(bestdice))
            Miou = miou / count
            Pa = pa / count
            torch.save(model.state_dict(), direc + "model.pth")
#输出验证结果
print("bestdice = {}".format(bestdice))
print("MIoU:", Miou)
#print("Pa:", Pa)

# 训练完成后，绘制 loss 曲线
epochs = np.arange(1, args.epochs + 1)  # 横轴表示 epoch 的整数索引
plt.plot(epochs, losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
# for i, loss in enumerate(losses):
#     plt.text(epochs[i], loss, f'{loss:.4f}', fontsize=9, ha='right', va='bottom')  # 标出每个点的 loss 值
plt.grid(True)
plt.show()
