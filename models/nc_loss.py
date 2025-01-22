import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAverage(nn.Module):
    def __init__(self, feat_dim, n_data, temp, momentum):
        super(LinearAverage, self).__init__()
        self.n_data = n_data
        self.mul = temp
        self.momentum = momentum
        self.register_buffer('memory', torch.randn(n_data, feat_dim).float())
        self.register_buffer('labels', torch.zeros(n_data).long())

    def update_weight(self, feat, index):
        with torch.no_grad():
            # update memory
            self.memory[index, :] = self.momentum * self.memory[index, :] + (1 - self.momentum) * feat
            self.memory[index, :] = F.normalize(self.memory[index, :])


    def forward(self, feat, index):
      
        feat = F.normalize(feat)
        ### Calculate mini-batch x memory similarity
        feat_mat = torch.matmul(feat, self.memory.t()) * self.mul
        return feat_mat

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
        feat_mat2.masked_fill_(mask, -1 / conf_model_temp)
        loss_nc = conf_train_eta * entropy(torch.cat([feat_mat,
                                                      feat_mat2], 1))
        return loss_nc

if __name__ == '__main__':
    # 예시 파라미터 설정
    feat_dim = 2048
    ndata = 1000 # 전체 타겟 데이터셋 크기
    temp = 0.05
    momentum = 0.9
    eta = 0.05
    batch_size = 100
    
    class Config():
      def __init__(self):
        self.temp = temp
    conf_model = Config()
    
    class TrainConfig():
      def __init__(self):
        self.eta = eta
        self.thr = 0.5
        self.margin = 1
    conf_train = TrainConfig()
    
    # 메모리 뱅크 초기화
    lemniscate = LinearAverage(feat_dim, ndata, conf_model.temp, conf_train.momentum).cuda()

    # 가상의 특징 벡터와 인덱스 생성
    feat_t = torch.randn(batch_size, feat_dim).cuda() # 미니배치 특징 벡터
    index_t = torch.randint(0, ndata, (batch_size,)).cuda()  # 미니 배치 특징 벡터에 해당하는 인덱스
    
    # NC 손실 계산
    loss_nc = calculate_nc_loss(feat_t, index_t, lemniscate, conf_model.temp, conf_train.eta, batch_size)

    print(f"Calculated loss_nc: {loss_nc}")