import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]
    if not target.data.dim():
        return torch.zeros(1)
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, size_average=True)
    return loss


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    return cross_entropy_2d(pred, label)


class Channel_Wise_DiffLoss(nn.Module):

    def __init__(self):
        super(Channel_Wise_DiffLoss, self).__init__()

    def forward(self, input1, input2):

        pixel_size = input1.size(2) * input1.size(3)
        input1 = input1.view(pixel_size, -1)
        input2 = input2.view(pixel_size, -1)
        '''
        pixel wise dim=1, channel wise dim=0
        '''
        input1_l2_norm = torch.norm(input1, p=2, dim=0, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=0, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        #trace = torch.trace((input1_l2.t().mm(input2_l2)))

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2))).pow(2)
        # diff_loss = torch.mean((input1_l2.permute(1, 0).mm(input2_l2)).pow(2))

        return diff_loss

class Pixel_Wise_DiffLoss(nn.Module):

    def __init__(self):
        super(Pixel_Wise_DiffLoss, self).__init__()

    def forward(self, input1, input2):

        pixel_size = input1.size(2) * input1.size(3)
        input1 = input1.view(pixel_size, -1)
        input2 = input2.view(pixel_size, -1)
        '''
        pixel wise dim=1, channel wise dim=0
        '''
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        # diff_loss = torch.mean((input1_l2.permute(1, 0).mm(input2_l2)).pow(2))

        return diff_loss
def least_squares_loss(feature, domain_label):
    return torch.mean((domain_label - feature) ** 2)

#
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True,sigmoid=False,reduce=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1) * 1.0
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce
    def forward(self, inputs, targets):
        N, C, H, W = inputs.size()
        if self.sigmoid:
            # 二分类
            P = F.sigmoid(inputs)
            #F.softmax(inputs)
            if targets == 0:
                probs = 1 - P#(P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            P = F.softmax(inputs, dim=1)
            class_mask = inputs.data.new(P.size()).fill_(0)
            ids = targets.view(N, 1, H, W)
            #class_mask = inputs.data.new(N, C).fill_(0)
            #ids = targets.view(-1, 1)  # 0 or 1 source or target
            class_mask.scatter_(1, ids.data, 1.)
            # print(class_mask)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            #alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(N, 1, H, W)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -1 * (torch.pow((1 - probs), self.gamma)) * log_p

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class FocalLoss_Reverse(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, sigmoid=False, reduce=True):
        super(FocalLoss_Reverse, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1) * 1.0
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce
    def forward(self, inputs, targets):
        N, C, H, W = inputs.size()
        if self.sigmoid:
            # 二分类
            P = F.sigmoid(inputs)
            #F.softmax(inputs)
            if targets == 0:
                probs = 1 - P#(P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            P = F.softmax(inputs, dim=1)
            class_mask = inputs.data.new(P.size()).fill_(0)
            ids = targets.view(N, 1, H, W)
            #class_mask = inputs.data.new(N, C).fill_(0)
            #ids = targets.view(-1, 1)  # 0 or 1 source or target
            class_mask.scatter_(1, ids.data, 1.)
            # print(class_mask)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            #alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(N, 1, H, W)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -1 * (torch.pow((probs), self.gamma)) * log_p

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss_Kaggle(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss_Kaggle, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.bce_loss = nn.BCEWithLogitsLoss()
    def forward(self, inputs, targets):
        BCE_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

if __name__ == '__main__':
    def conv3x3(in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    class netD(nn.Module):
        def __init__(self, inchannels, ndf=64):
            super(netD, self).__init__()
            self.feature = nn.Sequential(
                nn.Conv2d(inchannels, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
            )
            self.conv1 = nn.Conv2d(ndf * 8, 2, kernel_size=4, stride=2, padding=1)

        def forward(self, x):
            x = self.feature(x)
            x1 = self.conv1(x)
            # x1 = x1.view(-1, 1)
            return x1


    netd = netD(13)
    #focal_loss = FocalLoss(2, alpha=None, gamma=2, size_average=True, sigmoid=False, reduce=True)
    diff_loss = Pixel_Wise_DiffLoss()
    inputs_a = torch.randn(1, 13, 33, 65)
    inputs_b = inputs_a#torch.randn(1, 13, 33, 65)
    #aa, bb = netd(inputs_a), netd(inputs_b)
    loss = diff_loss(inputs_a, inputs_b)
    #loss = focal_loss(aa, torch.LongTensor(aa.size(0), 1, aa.size(2), aa.size(3)).fill_(0))
    print(loss)