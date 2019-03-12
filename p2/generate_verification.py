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
    checkpoints_path = "mnetv2_checkpoints2/"
    net = MNetV2.MNetV2()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    checkpoints_path = "verification_checkpoints/"

    net.load_state_dict(torch.load(checkpoints_path + "model_epoch0"))

    verification_val_path = "../hw2p2_check/validation_verification"
    val_order_file = "../hw2p2_check/validation_trials_verification.txt"
    img_list1, img_list2 = parse_verification_data(verification_val_path, val_order_file)
    val_veri_set = VerificationDataset(img_list1, img_list2)
    val_veri_loader = DataLoader(val_veri_set, shuffle=False)
    validate_verification(net, val_veri_loader, device, "val_verification.txt")

