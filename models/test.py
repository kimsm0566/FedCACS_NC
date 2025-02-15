import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y
from models.Nets import CNNCifar, CNNCifar100, CNNCifar_PFL, CNN_FEMNIST, MLP, get_vgg11
from models.new_Nets import cos_cifar100, cos_cifar10, ConvNet, ConvNet_wocos


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = np.int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label


class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label


def test_img_local(net_g, dataset, args, user_idx=-1, idxs=None, indd=None):
    net_g.eval()
    test_loss = 0
    correct = 0

    # put LEAF data into proper format
    if 'femnist' in args.dataset:
        leaf = True
        datatest_new = []
        usr = idx
        for j in range(len(dataset[usr]['x'])):
            datatest_new.append(
                (torch.reshape(torch.tensor(dataset[idx]['x'][j]), (1, 28, 28)), torch.tensor(dataset[idx]['y'][j])))
    elif 'sent140' in args.dataset:
        leaf = True
        datatest_new = []
        for j in range(len(dataset[idx]['x'])):
            datatest_new.append((dataset[idx]['x'][j], dataset[idx]['y'][j]))
    else:
        leaf = False

    if leaf:
        data_loader = DataLoader(DatasetSplit_leaf(datatest_new, np.ones(len(datatest_new))), batch_size=args.local_bs,
                                 shuffle=False)
    else:
        data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=False)
    if 'sent140' in args.dataset:
        hidden_train = net_g.init_hidden(args.local_bs)
    count = 0
    if len(data_loader.dataset) == 0: # <-- 이 부분을 추가하여 데이터셋이 비어있는 경우를 처리
      return 0, 0

    for idx, (data, target) in enumerate(data_loader):
        if 'sent140' in args.dataset:
            input_data, target_data = process_x(data, indd), process_y(target, indd)
            if args.local_bs != 1 and input_data.shape[0] != args.local_bs:
                break

            data, targets = torch.from_numpy(input_data).cuda(), torch.from_numpy(target_data).cuda()
            net_g.zero_grad()

            hidden_train = repackage_hidden(hidden_train)
            output, hidden_train = net_g(data, hidden_train)

            loss = F.cross_entropy(output.t(), torch.max(targets, 1)[1])
            _, pred_label = torch.max(output.t(), 1)
            correct += (pred_label == torch.max(targets, 1)[1]).sum().item()
            count += args.local_bs
            test_loss += loss.item()

        else:
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    if 'sent140' not in args.dataset:
        count = len(data_loader.dataset)
    if count == 0:
        return 0, 0
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return accuracy, test_loss


def test_img_local_all(net, args, dataset_test, dict_users_test, w_locals=None, w_glob_keys=None, indd=None,
                       dataset_train=None, dict_users_train=None, return_all=False):
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            a, b = test_img_local(net_local, dataset_test, args, idx=dict_users_test[idx], indd=indd, user_idx=idx)
            tot += len(dataset_test[dict_users_test[idx]]['x'])
        else:
            a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])
            tot += len(dict_users_test[idx])
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = a * len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = b * len(dataset_test[dict_users_test[idx]]['x'])
        else:
            acc_test_local[idx] = a * len(dict_users_test[idx])
            loss_test_local[idx] = b * len(dict_users_test[idx])
        del net_local

    if return_all:
        return acc_test_local, loss_test_local
    return sum(acc_test_local) / tot, sum(loss_test_local) / tot


def test_img(w_glob, args, datatest, return_probs=False):
    # net_glob = copy.deepcopy(net_g)
    if args.dataset == 'cifar10':
        if args.cos:
            net_g = cos_cifar10(args=args).cuda()
        else:
            net_g = CNNCifar(args=args).cuda()
    elif args.dataset == 'cifar100':
        if args.model == 'cnn':
            if args.cos:
                net_g = cos_cifar100(args=args).cuda()
            else:
                net_g = CNNCifar100(args=args).cuda()
        elif args.model == 'vgg':
            net_g = get_vgg11(100).cuda()
    elif args.dataset == 'mini_imagenet':
        opt = {'userelu': False, 'in_planes': 3, 'out_planes': [64, 64, 64, 64], 'num_stages': 4}
        if args.cos:
            net_g = ConvNet(opt).cuda()
        else:
            net_g = ConvNet_wocos(opt).cuda()
    elif args.dataset == 'femnist':
        net_g = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).cuda()
    else:
        net_g = CNNCifar_PFL(args=args).cuda()
    net_g.load_state_dict(w_glob)
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    probs = []

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            log_probs = net_g(data)

            # 추가: log_probs와 target 텐서 확인
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                print("Error: log_probs contains NaN or Inf values")
                continue
            if torch.isnan(target).any() or torch.isinf(target).any():
                print("Error: target contains NaN or Inf values")
                continue
            if target.dtype != torch.long and target.dtype != torch.int64: # 수정: torch.is_long, torch.is_int -> target.dtype
                 print("Error : Target is not long or int")
                 target = target.long()
           
            if (target.max() >= log_probs.shape[1]): # 이 부분에서 문제 발생
              print(f"target max: {target.max()}, log_probs_shape: {log_probs.shape}")
              target[target>= log_probs.shape[1]] = 0 # 수정: 레이블 값 범위를 벗어나는 경우 0으로 변경
           
        probs.append(log_probs)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    if return_probs:
        return accuracy, test_loss, torch.cat(probs)
    del net_g
    return test_loss, accuracy

def test_novel(args, net, dataset, idxs):
    net.eval()

    test_loss = 0
    correct = 0

    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=10, shuffle=True)
    # data_loader = DataLoader(dataset, batch_size=10)
    if idxs is None:
        length = len(dataset)
    else:
        length = len(idxs)

    probs = []

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            log_probs = net(data)
        probs.append(log_probs)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    # test_loss /= len(data_loader)
    # accuracy = 100.00 * float(correct) / len(data_loader)
    test_loss /= length
    accuracy = 100.00 * float(correct) / length

    del net
    return test_loss, accuracy
