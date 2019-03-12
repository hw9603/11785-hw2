import os
import time
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import MNetV1
import MNetV2
import ResNet
import face_classification
import score


class VerificationDataset(Dataset):
    def __init__(self, file_list1, file_list2):
        self.file_list1 = file_list1
        self.file_list2 = file_list2
        assert len(self.file_list1) == len(self.file_list2)

    def __len__(self):
        return len(self.file_list1)

    def __getitem__(self, index):
        # print(self.file_list[index])
        img1 = Image.open(self.file_list1[index])
        img1 = torchvision.transforms.ToTensor()(img1)
        img2 = Image.open(self.file_list2[index])
        img2 = torchvision.transforms.ToTensor()(img2)
        name1 = self.file_list1[index]
        name2 = self.file_list2[index]
        return img1, img2, name1, name2


def parse_verification_data(datadir, order_file):
    img_list1 = []
    img_list2 = []
    with open(order_file, "r") as f:
        for line in f.readlines():
            file1, file2, _ = line.replace('\n', '').split(" ")
            file1 = os.path.join(datadir, file1)
            file2 = os.path.join(datadir, file2)
            img_list1.append(file1)
            img_list2.append(file2)
    assert len(img_list1) == len(img_list2)
    print('{}:{}'.format('#Verification images pair', len(img_list1)))
    return img_list1, img_list2


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


def train(net, optimizer_label, optimizer_closs, criterion_label, criterion_closs, closs_weight,
          train_loader, val_loader, device, epoch):
    net.train()
    net.to(device)

    t = time.time()
    avg_loss = 0.0
    interval = 100
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # cuda
        inputs, labels = inputs.to(device), labels.to(device)
        # mini-batch GD
        optimizer_label.zero_grad()
        # optimizer_closs.zero_grad()

        feature, outputs = net.forward(inputs)

        l_loss = criterion_label(outputs, labels)
        # c_loss = criterion_closs(feature, labels)
        # loss = l_loss + closs_weight * c_loss
        loss = l_loss
        loss.backward()

        optimizer_label.step()
        # for param in criterion_closs.parameters():
        #     param.grad.data *= (1. / closs_weight)
        # optimizer_closs.step()

        avg_loss += loss.item()
        if batch_idx % interval == interval - 1:
            print("[Train Epoch %d] batch_idx=%d [%.2f%%, time: %.2f min], loss=%.4f" %
                  (epoch, batch_idx, 100. * batch_idx / len(train_loader), (time.time() - t) / 60, avg_loss / interval))
            avg_loss = 0.0
        if batch_idx % 2000 == 1999:
            print("Saving temporary checkpoints...")
            torch.save(net.state_dict(), "verification_checkpoints/model_epoch" + str(epoch))

    # train_loss, train_acc = test_classify(net, criterion_label, criterion_closs, closs_weight, train_loader, device)
    val_loss, val_acc = test_classify(net, criterion_label, criterion_closs, closs_weight, val_loader, device)
    # print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
    #       format(train_loss, train_acc, val_loss, val_acc))
    print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(val_loss, val_acc))


def test_classify(net, criterion_label, criterion_closs, closs_weight, test_loader, device):
    print("test classification loss...")
    net.eval()
    net.to(device)
    test_loss = []
    accuracy = 0
    total = 0

    interval = 1000
    t = time.time()

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = net.forward(feats)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        l_loss = criterion_label(outputs, labels.long())
        # c_loss = criterion_closs(feature, labels)
        # loss = l_loss + closs_weight * c_loss
        loss = l_loss

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels
        if batch_num % interval == interval - 1:
            print("batch_idx=%d [%.2f%%, time: %.2f min]" %
                  (batch_num, 100. * batch_num / len(test_loader), (time.time() - t) / 60))

    net.train()
    print('Test loss: {}, Accuracy: {}'.format(np.mean(test_loss), accuracy / total))
    return np.mean(test_loss), accuracy / total


def validate_verification(net, test_loader, device, output_file):
    net.eval()
    net.to(device)
    fwrite = open(output_file, "w")

    for feats1, feats2, name1, name2 in test_loader:
        feats1 = feats1.to(device)
        feats2 = feats2.to(device)
        feature1, _ = net.forward(feats1)
        feature2, _ = net.forward(feats2)
        distance = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = distance(feature1, feature2)
        for score in output.cpu().numpy():
            wstring = name1 + " " + name2 + " " + score + "\n"
            fwrite.write(wstring)
    fwrite.close()
    print("Verification file generated!")


def main():
    lr = 1e-4
    weight_decay = 1e-4
    batch_size = 128
    epochs = 4
    checkpoints_path = "mnetv2_checkpoints2/"
    net = MNetV2.MNetV2()
    num_classes = 4300
    feat_dim = 1280
    closs_weight = 1e-5
    lr_cent = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    net.load_state_dict(torch.load(checkpoints_path + "model_epoch7"))
    for param in net.parameters():
        param.requires_grad = False

    net.classifier = nn.Linear(1280, num_classes, bias=False)

    criterion_label = nn.CrossEntropyLoss()
    criterion_closs = CenterLoss(num_classes, feat_dim, device)

    optimizer_label = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_closs = torch.optim.Adam(criterion_closs.parameters(), lr=lr_cent)

    print("Load training data...")
    train_data_path_medium = "../hw2p2_check/train_data/medium"
    train_data_path_large = "../hw2p2_check/train_data/large"

    img_list_medium, label_list_medium = face_classification.parse_data(train_data_path_medium)
    img_list_large, label_list_large = face_classification.parse_data(train_data_path_large)
    img_list = img_list_medium + img_list_large
    label_list = label_list_medium + label_list_large
    trainset = face_classification.ImageDataset(img_list, label_list)
    for i in range(5):
        print(img_list[i], label_list[i])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    print("Load validation data...")
    val_data_path_medium = "../hw2p2_check/validation_classification/medium"
    val_data_path_large = "../hw2p2_check/validation_classification/large"
    img_list_medium, label_list_medium = face_classification.parse_data(val_data_path_medium)
    img_list_large, label_list_large = face_classification.parse_data(val_data_path_large)
    img_list = img_list_medium + img_list_large
    label_list = label_list_medium + label_list_large
    valset = face_classification.ImageDataset(img_list, label_list)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    train(net, optimizer_label, optimizer_closs, criterion_label, criterion_closs, closs_weight, train_loader, val_loader, device, epoch=0)
    model_path = "verification_checkpoints/" + "model_epoch0"
    optimizer_closs_path = "verification_checkpoints/" + "ocloss_epoch0"
    optimizer_label_path = "verification_checkpoints/" + "olabel_epoch0"
    torch.save(net.state_dict(), model_path)
    torch.save(optimizer_closs.state_dict(), optimizer_closs_path)
    torch.save(optimizer_label.state_dict(), optimizer_label_path)

    for param in net.parameters():
        param.requires_grad = True

    for epoch in range(epochs):
        train(net, optimizer_label, optimizer_closs, criterion_label, criterion_closs, closs_weight, train_loader, val_loader, device, epoch)
        model_path = "verification_checkpoints/" + "model_epoch" + str(epoch+1)
        optimizer_closs_path = "verification_checkpoints/" + "ocloss_epoch" + str(epoch+1)
        optimizer_label_path = "verification_checkpoints/" + "olabel_epoch" + str(epoch+1)
        torch.save(net.state_dict(), model_path)
        torch.save(optimizer_closs.state_dict(), optimizer_closs_path)
        torch.save(optimizer_label.state_dict(), optimizer_label_path)


if __name__ == "__main__":
    main()
