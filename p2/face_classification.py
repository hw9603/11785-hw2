import os
import time
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import MNetV1
import MNetV2


class ImageDataset(Dataset):
    def __init__(self, file_list, target_list=None, is_test=False):
        self.is_test = is_test
        if not is_test:
            self.file_list = file_list
            self.target_list = target_list
            self.n_class = len(list(set(target_list)))
        else:
            self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # print(self.file_list[index])
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        if not self.is_test:
            label = self.target_list[index]
            return img, label
        else:
            return img


def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]

    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n


def parse_test_classification_data(datadir, order_file):
    img_list = []
    with open(order_file, "r") as f:
        for line in f.readlines():
            filei = os.path.join(datadir, line.replace('\n', ''))
            img_list.append(filei)
    print('{}:{}'.format('#Test images', len(img_list)))
    return img_list


def train(net, optimizer, criterion, train_loader, val_loader, device, epoch, task='Classification'):
    net.train()
    net.to(device)

    t = time.time()
    avg_loss = 0.0
    interval = 100
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # cuda
        inputs, labels = inputs.to(device), labels.to(device)
        # mini-batch GD
        optimizer.zero_grad()
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        if batch_idx % interval == interval - 1:
            print("[Train Epoch %d] batch_idx=%d [%.2f%%, time: %.2f min], loss=%.4f" %
                  (epoch, batch_idx, 100. * batch_idx / len(train_loader), (time.time() - t) / 60, avg_loss / interval))
            avg_loss = 0.0

    if task == 'Classification':
        train_loss, train_acc = test_classify(net, criterion, train_loader, device)
        val_loss, val_acc = test_classify(net, criterion, val_loader, device)
        print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
              format(train_loss, train_acc, val_loss, val_acc))


def test_classify(net, criterion, test_loader, device):
    net.eval()
    net.to(device)
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = net(feats)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, labels.long())

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels

    net.train()
    print('Test loss: {}, Accuracy: {}'.format(np.mean(test_loss), accuracy / total))
    return np.mean(test_loss), accuracy / total


def classify_predict(net, test_loader, output_file, device):
    net.eval()
    net.to(device)
    fwrite = open(output_file, "w")
    fwrite.write("id,label\n")
    id = 0
    for feats in test_loader:
        feats = feats.to(device)
        outputs = net.forward(feats)
        # _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        _, pred_labels = torch.max(outputs.data, 1)
        pred_labels = pred_labels.view(-1)
        for label in pred_labels.cpu().numpy():
            print(id)
            wstring = str(id) + "," + str(label) + "\n"
            id += 1
            fwrite.write(wstring)
    fwrite.close()
    print("Prediction file generated!")


def main():
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 128
    epochs = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    print("Load training data...")
    train_data_path = "../hw2p2_check/train_data/medium"
    img_list, label_list, class_n = parse_data(train_data_path)
    trainset = ImageDataset(img_list, label_list)
    train_data_item, train_data_label = trainset.__getitem__(0)
    # print('data item shape: {}\t data item label: {}'.format(train_data_item.shape, train_data_label))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    print("Load validation data...")
    val_data_path = "../hw2p2_check/validation_classification/medium"
    img_list, label_list, class_n = parse_data(val_data_path)
    valset = ImageDataset(img_list, label_list)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    net = MNetV1.MNetV1()
    net.load_state_dict(torch.load("mnetv1_checkpoints/mnetv1_model_epoch40"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # print("Begin training...")
    for epoch in range(epochs):
        train(net, optimizer, criterion, train_loader, val_loader, device, epoch, task='Classification')
        if epoch % 5 == 1 or epoch == epochs - 1:
            path = "mnetv1model_epoch" + str(epoch)
            torch.save(net.state_dict(), path)
            torch.save(optimizer.state_dict(), )

    test_classify(net, criterion, val_loader, device)
    print("Load test data...")
    test_classification_data_path = "../hw2p2_check/test_classification/medium"
    test_classification_order_file = "../hw2p2_check/test_order_classification.txt"
    img_list = parse_test_classification_data(test_classification_data_path, test_classification_order_file)
    testset = ImageDataset(img_list, is_test=True)
    test_loader = DataLoader(testset)
    classify_predict(net, test_loader, "submission_classification.csv", device)


if __name__ == "__main__":
    main()