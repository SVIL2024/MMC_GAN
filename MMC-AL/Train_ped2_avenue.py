import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dataset_utils import resolve_split, split_dataset_by_video
#跑哪个数据集，切换哪个数据集的方法，上海数据集，用一个ped2的方法试试，看看效果怎么样，应该不会特别好，因为上海数据集的方法应用到了bkg背景信息，跑跑试试
import argparse


def validation_mse(model, loader):
    """Normal-only validation criterion; never reads test labels."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.cuda(non_blocking=True)
            prediction = model.forward(imgs[:, 0:12])[0]
            batch_size = imgs.size(0)
            total += F.mse_loss(prediction, imgs[:, -3:], reduction='mean').item() * batch_size
            count += batch_size
    if count == 0:
        raise RuntimeError('Validation loader is empty')
    return total / count

#
def MNADTrain():

    parser = argparse.ArgumentParser(description="DMAD")
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')

    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--phase1_epochs', type=int, default=6,
                        help='epochs trained without GAN before adversarial Phase 2 starts')
    parser.add_argument('--adversarial_weight', type=float, default=0.5,
                        help='alpha for the generator adversarial loss in Phase 2')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    #ped2的默认值是3，avenue的默认值是1，试试
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate') # why not 3e-4 XD
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for the train loader')
    parser.add_argument('--pin_memory', default=True, help='pinned memory for faster training, use more cpu')

    #
    #这个是ped2和avenue数据集的路径
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='dataset root or a directory containing video folders; auto-detected when omitted')

    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
    parser.add_argument('--log_type', type=str, default='realtime', help='type of log: txt, realtime')
    parser.add_argument('--resume', default=None, type=str,
                        help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
    parser.add_argument('--max_train_steps', type=int, default=None,
                        help='optional per-epoch step limit, useful for a smoke test')
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help='fraction of normal training videos reserved for model selection')
    parser.add_argument('--split_seed', type=int, default=2021,
                        help='seed for the video-level training/validation split')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='stop after this many non-improving validation epochs; 0 disables stopping')
    parser.add_argument('--skip_eval', '--skip_validation', dest='skip_validation', action='store_true',
                        help='use the fixed final epoch instead of validation-based model selection')


    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('This training script requires a CUDA GPU, but torch.cuda.is_available() is False.')
    if args.phase1_epochs < 0 or args.phase1_epochs > args.epochs:
        raise ValueError('--phase1_epochs must be between 0 and --epochs')

    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    train_folder = resolve_split(args.dataset_path, args.dataset_type, 'training')
    print('Training frames:', train_folder)

    #根据dataset_type数据集的不同，采用不用的加载方式
    train_dataset = DataLoader(str(train_folder), transforms.Compose([
                 transforms.ToTensor(),
                 ]), resize_height=args.h, resize_width=args.w, time_step=4, c=args.c,dataset_type = args.dataset_type)

    if args.skip_validation:
        train_subset = train_dataset
        val_subset = None
        train_videos = sorted(train_dataset.videos)
        val_videos = []
    else:
        train_subset, val_subset, train_videos, val_videos = split_dataset_by_video(
            train_dataset, args.val_fraction, args.split_seed)
    print('Training videos ({}): {}'.format(len(train_videos), train_videos))
    print('Validation videos ({}): {}'.format(len(val_videos), val_videos))

    train_batch = data.DataLoader(train_subset, batch_size = args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True, pin_memory=args.pin_memory)
    val_batch = None if val_subset is None else data.DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)

    # Model setting
    model = convAE_Unet_Sspcab(args.c, 5).cuda()
    optimizer = torch.optim.AdamW([{'params': model.unet_SSPCAB.parameters()},
        {'params': model.encoder.parameters()},
        {'params': model.offset_net.parameters()},
        ]
        , lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
    #对抗损失
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
    early_stop = {'idx': 0, 'best_val_loss': float('inf')}
    checkpoint_path = os.path.join(log_dir, 'bool_ped2.pth')
    log_interval = 100
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        phase2_active = epoch >= args.phase1_epochs
        if phase2_active:
            discriminator.train()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            print('Phase 2: joint adversarial training (alpha={})'.format(args.adversarial_weight))
        else:
            discriminator.eval()
            optimizer_D.zero_grad(set_to_none=True)
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            print('Phase 1: normality-constraint training (discriminator frozen, alpha=0)')

        for j,(imgs, _) in enumerate(train_batch):
            if args.max_train_steps is not None and j >= args.max_train_steps:
                break
            imgs = Variable(imgs).cuda() #imgs的torch.Size([1, 15, 256, 256])
            #model的输入是四张图片，这个outputs里面，有10个东西，里面有压缩信息，背景信息啥的
            outputs = model.forward(imgs[:,0:12])

            optimizer.zero_grad()

            if phase2_active:
                optimizer_D.zero_grad()
                D_l = discriminate_loss(discriminator(imgs[:,-3:]), discriminator(outputs[0].detach()))
                D_l.backward()
                optimizer_D.step()

                # Freeze D only while optimizing G, so adversarial gradients flow
                # through D to the generated frame without updating D twice.
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                g_l = adversarial_loss(discriminator(outputs[0]))
                loss = model.loss_function(imgs[:,-3:], *outputs) + args.adversarial_weight * g_l
            else:
                loss = model.loss_function(imgs[:,-3:], *outputs)

            loss.backward()
            optimizer.step()
            if phase2_active:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)

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

        if args.skip_validation:
            print('Validation skipped; the fixed final epoch will be saved')
            continue

        val_loss = validation_mse(model, val_batch)
        if val_loss < early_stop['best_val_loss']:
            early_stop['best_val_loss'] = val_loss
            early_stop['idx'] = 0
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'train_videos': train_videos,
                'val_videos': val_videos,
            }, checkpoint_path)
        else:
            early_stop['idx'] += 1
            print('Validation loss did not improve; model not saved')

        print('Epoch {} validation MSE: {:.8f}; best: {:.8f}; used time: {:.2f}s'.format(
            epoch + 1, val_loss, early_stop['best_val_loss'], time.time() - time_start))
        if args.early_stop_patience > 0 and early_stop['idx'] >= args.early_stop_patience:
            print('Early stopping after {} non-improving validation epochs'.format(early_stop['idx']))
            break

    if args.skip_validation:
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'selection_rule': 'fixed_final_epoch',
            'train_videos': train_videos,
            'val_videos': val_videos,
        }, checkpoint_path)

    print('Training is finished')
    sys.stdout = orig_stdout
    f.close()


if __name__=='__main__':
    MNADTrain()
