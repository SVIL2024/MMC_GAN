import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from tqdm import tqdm
from collections import OrderedDict
from model.utils import DataLoader
from model.final_future_prediction_shanghai import *
from utils import *
import glob
import argparse
from pathlib import Path
from dataset_utils import resolve_split

def MNADEval(model=None):
    parser = argparse.ArgumentParser(description="DMAD")
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='dataset root or a directory containing video folders')

    parser.add_argument('--model_dir', type=str,
                        default='./exp/shanghai/log/new_shanghai.pth',
                        help='checkpoint produced by Train_shanghai.py')

    args = parser.parse_args()

    torch.manual_seed(2020)
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
    test_folder = str(resolve_split(args.dataset_path, args.dataset_type, 'testing'))

    # Background templates are generated locally from their corresponding
    # split and are intentionally not stored in the source repository.
    if not glob.glob('./bkg/*'):
        train_folder = str(resolve_split(args.dataset_path, args.dataset_type, 'training'))
        os.makedirs('./bkg', exist_ok=True)
        gen_bkg(train_folder, './bkg/', (0, len(os.listdir(train_folder))))
    if not glob.glob('./bkg_/*'):
        os.makedirs('./bkg_', exist_ok=True)
        gen_bkg(test_folder, './bkg_/', (0, len(os.listdir(test_folder))))
    training_background_count = len(glob.glob('./bkg/*'))

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=args.h, resize_width=args.w, time_step=4, c=args.c,dataset_type = args.dataset_type)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    # Loading the trained model
    if model is None: # if not training, we give a exist model and params path
        model = convAE_me(args.c, 5)
        print("加载上海科技预训练模型:", args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location='cpu', weights_only=False)
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Backgrounds are data-dependent inference inputs, not learned weights.
        state_dict = {key: value for key, value in state_dict.items() if key != 'bkg'}
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected_missing = [key for key in incompatible.missing_keys if key != 'bkg']
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError('Incompatible checkpoint: missing={}, unexpected={}'.format(
                unexpected_missing, incompatible.unexpected_keys))
        # Keep the original ShanghaiTech evaluation protocol: after loading
        # the checkpoint, replace its trainable background parameter with the
        # 330 training and 107 testing templates from bkg/ and bkg_/.
        model.load_bkg(get_bkg(args.w, include_test=True))
        model.cuda()
    else:
        # A supplied model is evaluated directly; training no longer writes or
        # reloads a temporary checkpoint for per-epoch test evaluation.
        model.load_bkg(get_bkg(args.w, include_test=True))
        model.cuda()

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = os.path.basename(os.path.normpath(video))
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    # Frame masks use 0=normal and 1=anomaly.  Each prediction corresponds to
    # the fifth frame of a 5-frame window, so discard the first four labels of
    # every video to align labels with model scores.
    label_folder = str(Path(test_folder).parent / 'test_frame_mask')
    labels_list = np.concatenate([
        np.asarray(np.load(os.path.join(label_folder, video_name + '.npy'))).reshape(-1)[4:]
        for video_name in videos
    ]).astype(np.int64)

    label_length = 0
    list1 = {}
    list2 = {}
    list3 = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = os.path.basename(os.path.normpath(video))
        label_length += videos[video_name]['length']
        list1[video_name] = []
        list2[video_name] = []
        list3[video_name] = []
    
    label_length = 0
    video_num = 0
    label_length += videos[os.path.basename(os.path.normpath(videos_list[video_num]))]['length']

    model.eval()
    with torch.no_grad():
        for k,(imgs, view_idx) in enumerate(tqdm(test_batch)):

            if k == label_length-4*(video_num+1):
                video_num += 1
                next_video_name = os.path.basename(os.path.normpath(videos_list[video_num]))
                label_length += videos[next_video_name]['length']
            
            vidx = Variable((view_idx.reshape(-1) + training_background_count).long()).cuda()
            imgs = Variable(imgs).cuda()
            outputs = model.forward(imgs[:, 0:12],vidx, True)
            err_map = model.loss_function(imgs[:, -3:], vidx, *outputs, True)
            latest_losses = model.latest_losses()

            video_name = os.path.basename(os.path.normpath(videos_list[video_num]))
            list1[video_name].append(float(latest_losses['err1']))
            list2[video_name].append(float(latest_losses['err2_']))

            ##############norm with cv2.findContours################
            mask = outputs[7].squeeze(0).squeeze(0).cpu().numpy()
            mask = np.where(mask > 0.7, 1, 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max3 = 0
            for k in range(len(contours)):
                if cv2.contourArea(contours[k]) > 13:
                    mask_ = cv2.drawContours(np.zeros_like(mask), contours, k, 255, -1).astype(np.float32)
                    area = mask_.sum()
                    err = (mask_ * err_map.squeeze(0).cpu().numpy()).sum() / area
                    if err > max3:
                        max3 = err
            list3[video_name].append(max3)

    # Measuring the abnormality score and the AUC
    anomaly_list1 = []
    anomaly_list2 = []
    anomaly_list3 = []

    for video in sorted(videos_list):
        video_name = os.path.basename(os.path.normpath(video))
        anomaly_list1 += conf_avg(np.array(anomaly_score_list_inv(list1[video_name])), n_conf="robust+")
        anomaly_list2 += conf_avg(np.array(anomaly_score_list_inv(list2[video_name])), n_conf="robust+")
        anomaly_list3 += conf_avg(np.array(anomaly_score_list_inv(list3[video_name])), n_conf="robust+")

    os.makedirs("./exp/shanghai/log", exist_ok=True)
    np.savez("./exp/shanghai/log/res_list.npz", anomaly_list1, anomaly_list2, anomaly_list3, 1 - labels_list)

    anomaly_list1 = conf_avg(np.array(anomaly_list1), 55, "average")
    anomaly_list2 = conf_avg(np.array(anomaly_list2), 55, "average")
    anomaly_list3 = conf_avg(np.array(anomaly_list3), 55, "average")
    
    hyp_alpha = [0.2, 0.4, 0.6]

    comb = np.array(anomaly_list1) * hyp_alpha[0] + np.array(anomaly_list2) * hyp_alpha[1] + np.array(anomaly_list3) * hyp_alpha[2]
    accuracy = filter(np.array(conf_avg(comb, 55)), 1 - labels_list, test_folder)

    print('The result of ', args.dataset_type)
    print('AUC: ', accuracy * 100, '%; (alpha = ', hyp_alpha, ')')

    return accuracy * 100


if __name__=='__main__':
    MNADEval()

#     data = np.load("./exp.bak/res_list_shanghai.npz")

#     anomaly_list1 = conf_avg(np.array(data['arr_0']), 55, "average")
#     anomaly_list2 = conf_avg(np.array(data['arr_2']), 55, "average")
#     anomaly_list3 = conf_avg(np.array(data['arr_3']), 55, "average")

#     hyp_alpha = [0.2, 0.6, 0.4]
    
#     comb = np.array(anomaly_list1) * hyp_alpha[0] + np.array(anomaly_list2) * hyp_alpha[1] + np.array(anomaly_list3) * hyp_alpha[2]

#     accuracy = filter(np.array(conf_avg(comb, 55)), data['arr_5'][0], "..\\..\\shanghai\\testing\\frames")
#     print('AUC: ', accuracy * 100, '%; (alpha = ', hyp_alpha, ')')
