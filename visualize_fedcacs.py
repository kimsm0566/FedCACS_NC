import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from utils.options import args_parser
from models.Nets import CNNCifar, CNNCifar100, get_vgg11
from models.new_Nets import cos_cifar10, ConvNet_wocos, cos_cifar100
from models.Update import DatasetSplit
from utils.train_utils import get_data
from utils.mini_imagenet import get_imagenet_data
import time

def visualize_features(args, dataset_train, dataset_test, dict_users_train, dict_users_test):
    # 모델 준비
    if args.dataset == 'cifar10':
        if args.cos:
            net_g = cos_cifar10(args=args).cuda()
        else:
            net_g = CNNCifar(args=args).cuda()
    elif args.dataset == 'cifar100':
        if args.cos:
            net_g = cos_cifar100(args=args).cuda()
        else:
            net_g = CNNCifar100(args=args).cuda()
    elif args.model == 'vgg':
            net_g = get_vgg11(100).cuda()
    else:
        net_g = CNNCifar(args=args).cuda()
    net_g = net_g.eval() # 평가 모드 설정

    # 특징 추출 함수 (모델에서 특징 벡터 추출)
    def get_features(net, dataset, batch_size, idxs=None):
        features = []
        labels = []
        loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for data, target in loader:
                data = data.cuda()
                feat = net(data)
                features.append(feat.cpu().numpy())
                labels.append(target.cpu().numpy())

        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    # 특징 추출
    if 'mini_imagenet' in args.dataset:
      train_features, train_labels = get_features(net_g, dataset_train, args.bs)
    else:
       train_features, train_labels = get_features(net_g, dataset_train, args.bs, idxs=dict_users_train[0])  # CIFAR 데이터셋
    test_features, test_labels = get_features(net_g, dataset_test, args.bs, idxs=dict_users_test[0])
    all_features = np.concatenate((train_features, test_features), axis = 0)
    all_labels = np.concatenate((train_labels, test_labels), axis = 0)
    print("Start t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(all_features)
    
    print("t-SNE visualization finished.")
    
    # 색상 설정 (고유한 클래스 수에 따라 색상을 지정)
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)
    colors = plt.cm.get_cmap('hsv', num_classes)
    
    # 시각화
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = np.where(all_labels == label)
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors(label / num_classes), label=f'Class {label}')

    plt.title('t-SNE Visualization of Feature Vectors')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)

    # 저장 경로 설정
    output_path = f"./save/t-sne_{args.dataset}_{args.num_users}.png"  # 수정: 파일명에 데이터셋 이름을 포함
    plt.savefig(output_path)
    print(f"t-SNE plot saved to: {output_path}")

    plt.show()
    return tsne_results, all_labels


if __name__ == '__main__':
    # parse args
    args = args_parser()

    if 'cifar' in args.dataset or args.dataset == 'fmnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test, user_cat_index = get_data(args)

    else:
        if 'mini_imagenet' in args.dataset:
           dataset_train, dataset_test, dict_users_train, dict_users_test, user_cat_index = get_imagenet_data(args)

    tsne_results, all_labels = visualize_features(args, dataset_train, dataset_test, dict_users_train, dict_users_test)