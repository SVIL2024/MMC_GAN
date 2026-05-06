import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import copy
from collections import OrderedDict
from model.utils import DataLoader
from model.final_future_prediction_shanghai import *
from utils import *
import Evaluate_shanghai as Evaluate
import argparse
from losses import *
from model.pix2pix_networks import PixelDiscriminator
#下面是补充的两个创新点 ：这个创新点加入到下采样中的sspcab后面了
from WavePooling import WaveBlock
from HCFNet import PPA
def MNADTrain():
    parser = argparse.ArgumentParser(description="DMAD")

    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=1, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate') # why not 3e-4 XD
    parser.add_argument('--dim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=1000, help='number of the memory items')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--pin_memory', default=True, help='pinned memory for faster training, use more cpu')
    parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='E:\\dataset\\', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
    parser.add_argument('--log_type', type=str, default='realtime', help='type of log: txt, realtime')
    args = parser.parse_args()
    #随机种子
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    train_folder = args.dataset_path+args.dataset_type+"/training/frames"
    test_folder = args.dataset_path+args.dataset_type+"/testing/frames"
    #这个bkg是背景
    if "bkg" not in os.listdir('./'):
        os.mkdir("./bkg")
        gen_bkg(train_folder, "./bkg/")
        print("background images has been generated, please re-run the Train.py")
        return

    if "bkg_" not in os.listdir('./'):
        os.mkdir("./bkg_")
        gen_bkg(test_folder, "./bkg_/")
        print("background images has been generated, please re-run the Train.py")
        return

    bkg = get_bkg(args.w)
    print("loading a ",bkg.shape[0],"-views background template")

    # Loading dataset
    train_dataset = DataLoader(train_folder, transforms.Compose([
                 transforms.ToTensor(),
                 ]), resize_height=args.h, resize_width=args.w, time_step=4, c=args,dataset_type = args.dataset_type)
    train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True, pin_memory=args.pin_memory)

    # Model setting：这里调整为自己的模型
    model = convAE_me(args.c, 5, args.msize, args.dim, bkg=bkg)
    #初始化自己模型的参数
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': model.offset_net.parameters()},
        {'params': model.vq_layer.parameters()},
        {'params': model.bkg, "lr": 10*args.lr},]
        , lr=args.lr)

    #利用余弦退火方法调整学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.cuda()

    # GAN setting
    discriminator = PixelDiscriminator(input_nc=3).cuda()
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00002)

    # 对抗损失
    adversarial_loss = Adversarial_Loss().cuda()
    discriminate_loss = Discriminate_Loss().cuda()
    discriminator = discriminator.train().cuda()
    # 梯度损失
    gradient_loss = Gradient_Loss(3).cuda()
    # 强度损失
    intensity_loss = Intensity_Loss().cuda()

    # Report the training process
    log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    orig_stdout = sys.stdout
    f = open(os.path.join(log_dir, 'log.txt'),'w')
    if args.log_type == 'txt':
        sys.stdout = f

    # Training
    early_stop = {'idx' : 0,
                  'best_eval_auc' : 0}
    # record = [0]
    log_interval = 100
    #这个损失函数也得修改一下:这里就是用来展示训练过程的
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}

    for epoch in range(args.epochs):
        model.train()
        for j,(imgs, view_idx) in enumerate(train_batch):
            imgs = Variable(imgs).cuda()
            vidx = Variable((view_idx[:, 0,0,0]).long()).cuda()
            #这个模型也要修改一下
            outputs = model.forward(imgs[:,0:12], vidx)
            g_l = adversarial_loss(discriminator(outputs[0]))
            D_l = discriminate_loss(discriminator(imgs[:, -3:]), discriminator(outputs[0].detach()))
            # 梯度损失
            grad_l = gradient_loss(imgs[:, -3:], outputs[0])
            # 强度损失
            inte_l = intensity_loss(imgs[:, -3:], outputs[0])

            optimizer_D.zero_grad()
            D_l.backward()

            optimizer.zero_grad()

            loss = model.loss_function(imgs[:,-3:],vidx,  *outputs) + 0.05 * g_l + 0.3 * grad_l

            loss.backward()
            optimizer_D.step()
            optimizer.step()
            ########################################
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_train'] += float(latest_losses[key])
                epoch_losses[key + '_train'] += float(latest_losses[key])

            if j % log_interval == 0:
                for key in latest_losses:
                    losses[key + '_train'] /= log_interval
                loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
                print('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]  {loss}'
                             .format(epoch=epoch, batch=j * len(imgs),
                                     total_batch=len(train_batch) * len(imgs),
                                     percent=int(100. * j / len(train_batch)),
                                     loss=loss_string))
                print("best: ", early_stop['best_eval_auc'])
                for key in latest_losses:
                    losses[key + '_train'] = 0

        scheduler.step()

        print('----------------------------------------')
        print('Epoch:', epoch+1)

        print('----------------------------------------')
        torch.save(model, os.path.join(log_dir, 'temp.pth'))
        score = Evaluate.MNADEval(True)

        if score > early_stop['best_eval_auc']:
            early_stop['best_eval_auc'] = score
            early_stop['idx'] = 0
            torch.save(model, os.path.join(log_dir, 'new_shanghai.pth'))
        else:
            early_stop['idx'] += 1
            print('Score drop! Model not saved')
        print('With {} epochs, auc score is: {}, best score is: {}'.format(epoch+1, score,early_stop['best_eval_auc']))

    print('Training is finished')
    sys.stdout = orig_stdout
    f.close()


if __name__=='__main__':
    MNADTrain()
