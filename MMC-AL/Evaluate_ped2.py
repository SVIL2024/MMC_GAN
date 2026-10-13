import numpy as np
import os
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
from model.utils import DataLoader
from model.final_future_prediction_ped2 import *
from utils import *
import glob
import argparse
from pathlib import Path
from dataset_utils import load_ped2_labels, resolve_split
#这个目前可以评估ped2
def _load_checkpoint(model, model_path, device):
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            'MMC-GAN checkpoint not found: {}. Train first or pass --model_dir.'.format(path)
        )
    checkpoint = torch.load(str(path), map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        for key in ('net_g', 'model', 'state_dict'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    else:
        raise TypeError('Unsupported checkpoint type: {}'.format(type(checkpoint).__name__))

    model_keys = set(model.state_dict())
    matched = model_keys.intersection(state_dict)
    if not matched:
        raise ValueError(
            'Checkpoint {} is for a different network: none of its parameter names match MMC-GAN.'.format(path)
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    print('Loaded checkpoint:', path)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print('Checkpoint loaded non-strictly ({} missing, {} unexpected keys).'.format(
            len(incompatible.missing_keys), len(incompatible.unexpected_keys)))


def MNADEval(model=None, dataset_path=None, label_path=None):
    parser = argparse.ArgumentParser(description="DMAD")
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--num_workers_test', type=int, default=2, help='number of workers for the test loader')
    ####################
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='dataset root or a directory containing video folders; auto-detected when omitted')
    parser.add_argument('--label_path', type=str, default=None,
                        help='frame_labels_ped2.npy or the original Ped2 test.csv')

    # # 这个是上海科技的路径
    parser.add_argument('--model_dir', type=str, default='./exp/ped2/log/bool_ped2.pth',
                        help='checkpoint produced by Train_ped2_avenue.py')


    args, _ = parser.parse_known_args() if model is not None else (parser.parse_args(), [])
    if dataset_path is not None:
        args.dataset_path = dataset_path
    if label_path is not None:
        args.label_path = label_path
    # model_ped2_98.29.pth    convAE_Unet_Sspcab
    torch.manual_seed(2020)
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    # test_folder = args.dataset_path+args.dataset_type+"/testing/frames"
    test_folder = resolve_split(args.dataset_path, args.dataset_type, 'testing')

    print("test_folder",test_folder)

    #指定训练的数据集：不同数据集，加载数据的方式不一样
    test_dataset = DataLoader(str(test_folder), transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=args.h, resize_width=args.w, time_step=4, c=args.c,dataset_type = args.dataset_type)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    # Loading the trained model
    if model is None:  # if not training, we give a exist model and params path
        model = convAE_Unet_Sspcab(args.c, 5)
        _load_checkpoint(model, args.model_dir, torch.device('cuda'))
        model.cuda()

    labels_list = load_ped2_labels(test_folder, args.label_path, time_step=4)

    videos = OrderedDict()
    videos_list = [details['path'] for details in test_dataset.videos.values()]
    for video_name, details in test_dataset.videos.items():
        video = details['path']
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = details['frame']
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    label_length = 0
    list1 = {}
    list2 = {}
    list3 = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video_name in videos:
        label_length += videos[video_name]['length']
        list1[video_name] = []
        list2[video_name] = []
        list3[video_name] = []

    video_names = list(videos)
    print("videos_list", videos_list)

    model.eval()
    with torch.no_grad():
        for k,(imgs, video_idx) in enumerate(test_batch):
            imgs = Variable(imgs).cuda()

            outputs = model.forward(imgs[:, 0:12],True)
            model.loss_function(imgs[:, -3:], *outputs, True)
            latest_losses = model.latest_losses()

            if len(video_idx) != 1:
                raise ValueError('Evaluation currently requires --test_batch_size 1')
            video_name = video_names[int(video_idx.item())]
            list1[video_name].append(float(latest_losses['err1']))
            list2[video_name].append(float(latest_losses['mse2']))
            list3[video_name].append(float(latest_losses['grad']))

    # Measuring the abnormality score and the AUC
    anomaly_list1 = []
    anomaly_list2 = []
    anomaly_list3 = []

    for video_name in videos:
        anomaly_list1 += anomaly_score_list_inv(list1[video_name])
        anomaly_list2 += anomaly_score_list_inv(list2[video_name])
        anomaly_list3 += anomaly_score_list_inv(list3[video_name])
    #这里别忘记修改
    result_dir = Path('./exp') / args.dataset_type / 'log'
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(result_dir / 'res_list.npz'), anomaly_list1, anomaly_list2, anomaly_list3, labels_list)
    # np.savez("./exp/shanghai/log/res_list1.npz", anomaly_list1, anomaly_list2, anomaly_list3, 1 - labels_list)

    def conf_avg(x, size=11, n_conf=5):
        a = x.copy()
        b = []
        weight = np.array([1, 1, 1, 1, 1.2, 1.6, 1.2, 1, 1, 1, 1])

        for i in range(x.shape[0] - size + 1):
            a_ = a[i:i + size].copy()
            u = a_.mean()
            dif = abs(a_ - u)
            sot = np.argsort(dif)[:n_conf]
            mask = np.zeros_like(dif)
            mask[sot] = 1
            weight_ = weight * mask
            b.append(np.sum(a_ * weight_) / weight_.sum())
        for _ in range(size // 2):
            b.append(b[-1])
            b.insert(0, 1)
        return b

    anomaly_list1 = conf_avg(np.array(anomaly_list1))
    anomaly_list2 = conf_avg(np.array(anomaly_list2))
    anomaly_list3 = conf_avg(np.array(anomaly_list3))
    #这个是目前效果最好的参数，但是在可视化的时候，会导致有的异常分数大于1
    hyp_alpha = [0.2, 0.3, 0.5]



    #三个异常得分，合成的异常得分
    comb = np.array(anomaly_list1) * hyp_alpha[0] + np.array(anomaly_list2) * hyp_alpha[1] + np.array(anomaly_list3) * hyp_alpha[2]
    if len(comb) != len(labels_list):
        raise ValueError('Score/label length mismatch: {} scores versus {} labels'.format(
            len(comb), len(labels_list)))
    accuracy = roc_auc_score(y_true=labels_list, y_score=comb)

    print('The result of ', args.dataset_type)
    print('AUC: ', accuracy*100, '%; (alpha = ', hyp_alpha, ')')
    #这里稍微修改了一下，多加了后面两个参数，用来可视化的
    return accuracy*100,labels_list,comb

if __name__=='__main__':
    AUC,label,score = MNADEval()



    # for i in range(len(score)):
    #     print(f'id{i} score{score} label {label}')

    #torch.save(y1, './result/score0.csv')
    # torch.save(y2, './result/label.csv')
    # label_list = []
    # for sublist in labels_list:
    #     for element in sublist:
    #         label_list.append(element)
    # print(anomaly_score_total_list.shape)
    # print(label_list)


    #对测试时，得到的score和lable进行可
    score = score.squeeze()
    label = label.squeeze()

    x = torch.arange(len(score)).tolist()
    y1 = score
    y2 = 1 - label
    plt.figure(figsize=(12, 4))
    plt.plot(x, y1, label='anomaly score')
    plt.plot(x, label, label='anomalous event')
    plt.xlabel('frame', fontsize=15)
    plt.ylabel('anomaly score', fontsize=15)
    plt.legend(fontsize='xx-small')
    output_path = Path('./exp/ped2/log/model_05.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    # plt.savefig('exp_up56\\ped2\\log_lr0.001_weight_recon\\model_05.png')
    plt.show()

#     data = np.load("./exp.bak/res_list_ped2.npz")
#     def conf_avg(x, size=11, n_conf=5):
#         a = x.copy()
#         b = []
#         weight = np.array([1, 1, 1, 1, 1.2, 1.6, 1.2, 1, 1, 1, 1])

#         for i in range(x.shape[0] - size + 1):
#             a_ = a[i:i + size].copy()
#             u = a_.mean()
#             dif = abs(a_ - u)
#             sot = np.argsort(dif)[:n_conf]
#             mask = np.zeros_like(dif)
#             mask[sot] = 1
#             weight_ = weight * mask
#             b.append(np.sum(a_ * weight_) / weight_.sum())
#         for _ in range(size // 2):
#             b.append(b[-1])
#             b.insert(0, 1)
#         return b

#     anomaly_list1 = conf_avg(np.array(data['arr_0']))
#     anomaly_list2 = conf_avg(np.array(data['arr_1']))
#     anomaly_list3 = conf_avg(np.array(data['arr_4']))

#     hyp_alpha = [0.2, 0.4, 0.6]

#     comb = np.array(anomaly_list1) * hyp_alpha[0] + np.array(anomaly_list2) * hyp_alpha[1] + np.array(anomaly_list3) * hyp_alpha[2]
#     accuracy = roc_auc_score(y_true=data['arr_5'][0], y_score=comb)

#     print('AUC: ', accuracy*100, '%; (alpha = ', hyp_alpha, ')')
