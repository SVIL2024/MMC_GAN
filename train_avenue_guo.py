import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader,weights_init_normal

from model.final_future_prediction_ped2 import *
from utils import *
from losses import *
from model.pix2pix_networks import PixelDiscriminator
# import Evaluate_ped2  as Evaluate
import Evaluate_avenue_guo as Evaluate
import argparse

def MNADTrain():
    parser = argparse.ArgumentParser(description="DMAD")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate') # why not 3e-4 XD
    parser.add_argument('--dim', type=int, default=256, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=200, help='number of the memory items')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--pin_memory', default=True, help='pinned memory for faster training, use more cpu')
    parser.add_argument('--dataset_type', type=str, default='avenue', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='E:\\DMAD-master\\DMAD-PDM\\data\\datasets\\', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
    parser.add_argument('--log_type', type=str, default='realtime', help='type of log: txt, realtime')
    parser.add_argument('--resume', default=None, type=str,
                        help='The pre-trained model to resume training with, pass \'latest\' or the model name.')


    args = parser.parse_args()

    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    train_folder = args.dataset_path+args.dataset_type+"/Train/frames"

    # Loading dataset
    train_dataset = DataLoader(train_folder, transforms.Compose([
                 transforms.ToTensor(),
                 ]), resize_height=args.h, resize_width=args.w, time_step=4, c=args.c)

    train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True, pin_memory=args.pin_memory)

    # Model setting
    model = convAE_Unet_Sspcab(args.c, 5, args.msize, args.dim)
    optimizer = torch.optim.AdamW([{'params': model.unet_SSPCAB.parameters()},
        # {'params': model.decoder.parameters()},
        {'params': model.offset_net.parameters()},
        {'params': model.vq_layer.parameters()},
        {'params': model.bkg, "lr": 50*args.lr},]
        , lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.cuda()

    # GAN setting
    discriminator = PixelDiscriminator(input_nc=3).cuda()

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00002)

    if args.resume:
        discriminator.load_state_dict(torch.load(args.resume)['net_d'])
        optimizer_D.load_state_dict(torch.load(args.resume)['optimizer_d'])
        # 如果引用预训练模型，在这里调整学习率
        optimizer_D.param_groups[0]['lr'] = 0.00002  # 直接修改参数组的lr
        print(f'Pre-trained discriminator have been loaded.\n')
    else:
        discriminator.apply(weights_init_normal)
        print('Generator and discriminator are going to be trained from scratch.\n')

    adversarial_loss = Adversarial_Loss().cuda()
    discriminate_loss = Discriminate_Loss().cuda()
    discriminator = discriminator.train().cuda()







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
    log_interval = 100
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        for j,(imgs, _) in enumerate(train_batch):
            imgs = Variable(imgs).cuda() #imgs的torch.Size([1, 15, 256, 256])
            #model的输入是四张图片，这个outputs里面，有10个东西，里面有压缩信息，背景信息啥的
            outputs = model.forward(imgs[:,0:12])

            g_l = adversarial_loss(discriminator(outputs[0]))
            D_l = discriminate_loss(discriminator(imgs[:,-3:]), discriminator(outputs[0].detach()))
            optimizer_D.zero_grad()
            D_l.backward()
            optimizer.zero_grad()
            # 这里加上了一个g_l损失
            loss = model.loss_function(imgs[:,-3:], *outputs) + 0.05 * g_l
            loss.backward()
            optimizer_D.step()
            optimizer.step()

            ########################################
            #损失记录和展示
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_train'] += float(latest_losses[key])
                epoch_losses[key + '_train'] += float(latest_losses[key])

            if j % log_interval == 0:
                for key in latest_losses:
                    losses[key + '_train'] /= log_interval
                loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
                print('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                             ' {time:3.2f}   {loss} '
                             .format(epoch=epoch, batch=j * len(imgs),
                                     total_batch=len(train_batch) * len(imgs),
                                     percent=int(100. * j / len(train_batch)),
                                     time=time.time() - start_time,
                                     loss=loss_string),
                                     )
                start_time = time.time()
                for key in latest_losses:
                    losses[key + '_train'] = 0

        scheduler.step()
        if epoch>4:optimizer.param_groups[-1]['lr'] = args.lr*20
        print('----------------------------------------')
        print('Epoch:', epoch+1, '; Time:', time.time()-start_time)
        print('----------------------------------------')

        time_start = time.time()

        score = Evaluate.MNADEval(model=model)

        if score > early_stop['best_eval_auc']:
            early_stop['best_eval_auc'] = score
            early_stop['idx'] = 0
            torch.save(model, os.path.join(log_dir, f'model_Avenue.pth'))
        else:
            early_stop['idx'] += 1
            print('Score drop! Model not saved')

        print('With {} epochs, auc score is: {}, best score is: {}, used time: {}'.format(epoch+1, score, early_stop['best_eval_auc'], time.time()-time_start))


    print('Training is finished')

    sys.stdout = orig_stdout
    f.close()


if __name__=='__main__':
    MNADTrain()
