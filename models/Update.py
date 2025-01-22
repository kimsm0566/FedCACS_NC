import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import time
import copy

from models.test import test_img_local
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y
from models.LinearAverage import LinearAverage


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[np.int(self.idxs[item])]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]), (1, 28, 28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.idxs = idxs
        self.indd = None
        if idxs is not None and len(idxs) > 0:
           self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = [] # 수정: 빈 DataLoader로 초기화

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1):
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )

        local_eps = self.args.local_ep
        if last:
            if self.args.alg == 'fedavg' or self.args.alg == 'prox':
                local_eps = 10
                net_keys = [*net.state_dict().keys()]
                if 'cifar' in self.args.dataset:
                    w_glob_keys = [net.weight_keys[i] for i in [0, 1, 3, 4]]
                elif 'sent140' in self.args.dataset:
                    w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
                elif self.args.dataset == 'mini_imagenet':
                    if self.args.cos:
                        w_glob_keys = net_keys[2:]
                    else:
                        w_glob_keys = net_keys[:-2]
            local_eps = max(10, local_eps - self.args.local_second_ep)

        head_eps = local_eps - self.args.local_second_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False

            if (iter < head_eps and self.args.alg == 'fedrep') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            elif iter >= head_eps and self.args.alg == 'fedrep' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).cuda(), torch.from_numpy(target_data).to(
                        self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                else:
                    images, labels = images.cuda(), labels.cuda()
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            if len(batch_loss) > 0:
              epoch_loss.append(sum(batch_loss) / len(batch_loss))
            else:
              epoch_loss.append(0) # 수정: batch_loss가 빈 리스트인 경우 0을 추가

            if done:
                break
        if len(epoch_loss) > 0:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd
        else:
            return net.state_dict(), 0, self.indd

def entropy(x):
    return -torch.mean(torch.sum(F.softmax(x, dim=1) * F.log_softmax(x, dim=1), 1))

def calculate_nc_loss(feat_t, index_t, lemniscate, conf_model_temp, conf_train_eta, batch_size):
    ### Calculate mini-batch x memory similarity
    feat_mat = lemniscate(feat_t, index_t)
    ### We do not use memory features present in mini-batch
    feat_mat[:, index_t] = -1 / conf_model_temp
    ### Calculate mini-batch x mini-batch similarity
    feat_mat2 = torch.matmul(feat_t,
                                 feat_t.t()) / conf_model_temp
    mask = torch.eye(feat_mat2.size(0),
                            feat_mat2.size(0)).bool().cuda()
    feat_mat2.masked_fill_(mask, -1 / conf.model.temp)
    loss_nc = conf_train_eta * entropy(torch.cat([feat_mat,
                                                      feat_mat2], 1))
    return loss_nc

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
         d = np.int(self.idxs[item])
         image, label = self.dataset[d]
         return image, label

class LocalUpdateCACS_NC(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.dataset = dataset
        self.idxs = idxs
        if indd is not None:
            self.indd = indd
        else:
            self.indd = None
        if idxs is not None and len(idxs) > 0:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = []  # 수정: 빈 DataLoader로 초기화

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1):
        # print("start train")
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )

        local_eps = self.args.local_ep
        if last:
            local_eps = max(10, local_eps - self.args.local_second_ep)

        rep_eps = local_eps - self.args.local_second_ep
        epoch_loss = []
        num_updates = 0
        if 'mini_imagenet' in self.args.dataset: # Mini-ImageNet 데이터셋에 대해 메모리 뱅크 생성
            ndata = len(self.ldr_train.dataset) # 타겟 데이터셋 크기
            lemniscate = LinearAverage(2048, ndata, self.args.temp, self.args.momentum).cuda()
        else:
           lemniscate = None
        for iter in range(local_eps):
            done = False

            if (iter < rep_eps and self.args.alg == 'fedcacsNC') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            if (iter < rep_eps and self.args.alg == 'fedcacsNC') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.cuda(), labels.cuda()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                if iter >= rep_eps and self.args.alg == 'fedcacs' and not last and lemniscate is not None:
                    print("NC")
                    feat_t = net.get_features(images)
                    index_t = torch.arange(len(images)).cuda()
                    loss_nc = calculate_nc_loss(feat_t, index_t, lemniscate, self.args.temp, self.args.eta, self.args.local_bs)
                    loss = 0.5 * loss + 0.5 * loss_nc  # 수정: 가중 합 적용
                    lemniscate.update_weight(feat_t, index_t)
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            if len(batch_loss) > 0:
              epoch_loss.append(sum(batch_loss) / len(batch_loss))
            else:
              epoch_loss.append(0)
            if done:
                break
        if len(epoch_loss) > 0 :
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd
        else:
            return net.state_dict(), 0 , self.indd


# paper original
class LocalUpdateCACS(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.dataset = dataset
        self.idxs = idxs
        if indd is not None:
            self.indd = indd
        else:
            self.indd = None
        if idxs is not None and len(idxs) > 0:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = []  # 수정: 빈 DataLoader로 초기화
    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1):
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )

        local_eps = self.args.local_ep
        if last:
            local_eps = max(10, local_eps - self.args.local_second_ep)

        rep_eps = local_eps - self.args.local_second_ep
        epoch_loss = []
        num_updates = 0
        for iter in range(local_eps):
            done = False

            if (iter < rep_eps and self.args.alg == 'fedcacs') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif iter >= rep_eps and self.args.alg == 'fedcacs' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.cuda(), labels.cuda()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            if len(batch_loss) > 0:
              epoch_loss.append(sum(batch_loss) / len(batch_loss))
            else:
              epoch_loss.append(0)
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if len(epoch_loss) > 0 :
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd
        else:
            return net.state_dict(), 0 , self.indd