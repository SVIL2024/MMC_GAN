import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import copy
import glob
from collections import OrderedDict
from model.utils import DataLoader
from model.final_future_prediction_shanghai import *
from utils import *
import argparse
from losses import *
from model.pix2pix_networks import PixelDiscriminator
from dataset_utils import resolve_split, split_dataset_by_video


def validation_mse(model, loader):
    """Select checkpoints on held-out normal training videos only."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for imgs, view_idx in loader:
            imgs = imgs.cuda(non_blocking=True)
            vidx = view_idx.reshape(-1).long().cuda(non_blocking=True)
            prediction = model.forward(imgs[:, 0:12], vidx, True)[0]
            batch_size = imgs.size(0)
            total += F.mse_loss(prediction, imgs[:, -3:], reduction='mean').item() * batch_size
            count += batch_size
    if count == 0:
        raise RuntimeError('Validation loader is empty')
    return total / count


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
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate') # why not 3e-4 XD
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--pin_memory', default=True, help='pinned memory for faster training, use more cpu')
    parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='dataset root or a directory containing video folders')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
    parser.add_argument('--log_type', type=str, default='realtime', help='type of log: txt, realtime')
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
    #随机种子
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    train_folder = resolve_split(args.dataset_path, args.dataset_type, 'training')
    #这个bkg是背景
    if not os.path.isdir('./bkg') or not glob.glob('./bkg/*'):
        os.makedirs("./bkg", exist_ok=True)
        gen_bkg(train_folder, "./bkg/", (0, len(os.listdir(train_folder))))
        print("Training background templates have been generated.")

    bkg = get_bkg(args.w, include_test=False)
    print("loading a ",bkg.shape[0],"-views background template")

    # Loading dataset
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

    # Model setting：这里调整为自己的模型
    model = convAE_me(args.c, 5, bkg=bkg).cuda()
    #初始化自己模型的参数
    optimizer = torch.optim.AdamW([{'params': model.unet_SSPCAB.parameters()},
        {'params': model.encoder.parameters()},
        {'params': model.offset_net.parameters()},
        ]
        , lr=args.lr)

    #利用余弦退火方法调整学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # GAN setting
    discriminator = PixelDiscriminator(input_nc=3).cuda()
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00002)

    # 对抗损失
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
    checkpoint_path = os.path.join(log_dir, 'new_shanghai.pth')
    # record = [0]
    log_interval = 100
    #这个损失函数也得修改一下:这里就是用来展示训练过程的
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}

    for epoch in range(args.epochs):
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

        for j,(imgs, view_idx) in enumerate(train_batch):
            if args.max_train_steps is not None and j >= args.max_train_steps:
                break
            imgs = Variable(imgs).cuda()
            vidx = Variable(view_idx.reshape(-1).long()).cuda()
            #这个模型也要修改一下
            outputs = model.forward(imgs[:,0:12], vidx)
            optimizer.zero_grad()

            if phase2_active:
                optimizer_D.zero_grad()
                D_l = discriminate_loss(discriminator(imgs[:, -3:]), discriminator(outputs[0].detach()))
                D_l.backward()
                optimizer_D.step()

                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                g_l = adversarial_loss(discriminator(outputs[0]))
                loss = model.loss_function(imgs[:,-3:], vidx, *outputs) + \
                    args.adversarial_weight * g_l
            else:
                loss = model.loss_function(imgs[:,-3:], vidx, *outputs)

            loss.backward()
            optimizer.step()
            if phase2_active:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
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
                print("best validation MSE: ", early_stop['best_val_loss'])
                for key in latest_losses:
                    losses[key + '_train'] = 0

        scheduler.step()

        print('----------------------------------------')
        print('Epoch:', epoch+1)

        print('----------------------------------------')
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
        print('Epoch {} validation MSE: {:.8f}; best: {:.8f}'.format(
            epoch + 1, val_loss, early_stop['best_val_loss']))
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
