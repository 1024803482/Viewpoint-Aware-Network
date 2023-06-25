import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import os

from PolypDataset import PolypDataset, ValDataset, TestDataset
from Classifier import ResNet50, ResNeXt50, ResNet18
from utils import cal_cls_acc, AvgMeter


def train(train_loader,
          model,
          optimizer,
          epoch,
          batch_size=16,
          total_epoch=100,
          save_name="ResNet18_0212",
          grad_norm=5.0):
    model.train()
    cls_loss_record = AvgMeter()
    total_step = len(train_loader)
    for step, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, labels, masks = pack

        images = Variable(images).cuda()
        labels = Variable(labels).long().cuda()
        # masks = Variable(masks).float().cuda()

        # ---- forward ----
        r = model(images)

        # ---- calculate loss ----
        loss = F.cross_entropy(input=r, target=labels)

        # ---- loss backward ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ---- record ----
        cls_loss_record.update(loss.data, batch_size)

        # ---- train visualization ----
        if step % 10 == 0 or step == total_step:
            print("------ {} ------".format(datetime.now()))
            print("Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [CE_Loss: {:.4f}]".format(epoch, total_epoch, step, total_step, cls_loss_record.show()))

    save_path = "/data2/clh/workspace/OANet/save_model/{}/".format(save_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path, exist_ok=True)

    if epoch % 3 == 0:
        torch.save(model.state_dict(), save_path + 'epoch_%d.pth' % epoch)
    return True


def val(model,
        best_acc,
        val_loader,
        save_flag=False,
        save_name='ResNet34_0212',
        val_flag = 'val'):
    model.eval()

    loss_bank = []
    cls_acc_bank = []
    for step, pack in enumerate(val_loader, start=1):
        # ---- data prepare ----
        image, label, mask = pack
        image = image.cuda()
        label = label.long().cuda()
        # mask = mask.cuda()
        with torch.no_grad():
            r = model(image)
        # ---- calculate loss ----
        loss = F.cross_entropy(input=r, target=label)
        loss = loss.item()

        cls_acc = cal_cls_acc(r, label)

        loss_bank.append(loss)
        cls_acc_bank.append(cls_acc.cpu().numpy())

    print('{} Loss: {:.4f},  Acc: {:.4f}'.
          format(val_flag, np.mean(loss_bank), np.mean(cls_acc_bank)))

    if save_flag and best_acc < np.mean(cls_acc_bank):
        save_path = '/data2/clh/workspace/OANet/save_model/{}/'.format(save_name)
        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), save_path + 'Best_Acc.pth')
    return np.mean(cls_acc_bank)


if __name__ == "__main__":
    model = ResNet18(in_channels=3, num_classes=2, pretrained=True)
    model = model.cuda()
    params = model.parameters()
    print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    total_epoch = 50
    batch_size = 64
    seed = 205
    grad_norm = 10.0
    base_lr = 1e-3
    save_name = 'ResNet18_0323_pretrained'
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    dataset = PolypDataset(root_path='./Datasets/VANet_Dataset/TrainDataset/',
                           shape=(352, 352))
    val_dataset = ValDataset(root_path='./Datasets/VANet_Dataset/TrainDataset/',
                             shape=(352, 352))

    indices = list(range(len(dataset)))
    split = int(np.floor(0.2 * len(dataset)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    val_indices = indices[:split]
    train_indices = indices[split:]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             sampler=valid_sampler, num_workers=4, pin_memory=True)
    total_step = len(train_loader)
    best_acc = 0.0

    print("*" * 20, "|| Start Training ||", "*" * 20)
    for epoch in range(1, total_epoch + 1):
        optimizer.zero_grad()
        train(train_loader,
              model,
              optimizer,
              epoch,
              batch_size=batch_size,
              total_epoch=total_epoch,
              save_name=save_name,
              grad_norm=grad_norm)
        lr_ = base_lr * (1.0 - epoch / total_epoch) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        val_acc = val(model,
                      best_acc,
                      val_loader,
                      save_flag=True,
                      save_name=save_name)
        if val_acc > best_acc:
            best_acc = val_acc
            print("best acc. is {:.4f}".format(val_acc))