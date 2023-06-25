import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import os

from lib.models import QuickGELU, LayerNorm
from functools import partial
from PolypDataset import PolypDataset, ValDataset
from VANet import VANet
from utils import structure_loss, mean_dice_np, mean_iou_np, mean_seg_acc_np, AvgMeter

from torchcam.methods import GradCAMpp, SmoothGradCAMpp
from Classifier import ResNet18, ResNet50


def train(train_loader,
          model,
          optimizer,
          epoch,
          classifier,
          cam_extractor,
          batch_size=16,
          total_epoch=100,
          save_name="Unet0210",
          grad_norm=5.0):
    model.train()
    seg_loss_record = AvgMeter()
    total_step = len(train_loader)
    for step, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, labels, masks = pack

        images = Variable(images).cuda()
        # labels = Variable(labels).cuda()
        masks = Variable(masks).float().cuda()
        cue = None
        out = classifier(images)
        none_map = cam_extractor(1, out)[0]  # b h w
        if torch.sum(torch.isnan(none_map)) > 0:
            print("nan exists")
        else:
            cue = none_map.unsqueeze(1)

        # ---- forward ----
        r3, r2, r1, r0 = model(images, cue)

        r3 = F.interpolate(r3, (352, 352), mode='bilinear', )
        r2 = F.interpolate(r2, (352, 352), mode='bilinear', )
        r1 = F.interpolate(r1, (352, 352), mode='bilinear', )
        r0 = F.interpolate(r0, (352, 352), mode='bilinear', )

        # ---- calculate loss ----
        loss3 = structure_loss(masks, r3, kernel_size=31, weights=5.0)
        loss2 = structure_loss(masks, r2, kernel_size=31, weights=5.0)
        loss1 = structure_loss(masks, r1, kernel_size=31, weights=5.0)
        loss0 = structure_loss(masks, r0, kernel_size=31, weights=5.0)

        loss = 0.3 * loss3 + 0.3 * loss2 + 0.2 * loss1 + 0.2 * loss0

        # ---- loss backward ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ---- record ----
        seg_loss_record.update(loss.data, batch_size)

        # ---- train visualization ----
        if step % 10 == 0 or step == total_step:
            print("------ {} ------".format(datetime.now()))
            print(
                "Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [Structure_loss: {:.4f}]".format(epoch, total_epoch, step,
                                                                                               total_step,
                                                                                               seg_loss_record.show()))

    save_path = "./save_model/{}/".format(save_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path, exist_ok=True)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), save_path + 'epoch_%d.pth' % epoch)
    return True


def val(model,
        best_dice,
        val_loader,
        classifier,
        cam_extractor,
        epoch=1,
        save_flag=False,
        save_name='Unet0211',
        val_flag='val'):
    model.eval()
    dice_bank = []
    iou_bank = []
    loss_bank = []
    seg_acc_bank = []
    for step, pack in enumerate(val_loader, start=1):
        # ---- data prepare ----
        image, label, mask = pack
        image = image.cuda()
        # label = label.cuda()
        mask = mask.cuda()
        cue = None
        out = classifier(image)
        none_map = cam_extractor(1, out)[0]  # b h w
        if torch.sum(torch.isnan(none_map)) > 0:
            print("nan exists")
        else:
            cue = none_map.unsqueeze(1)
        with torch.no_grad():
            r3, r2, r1, r0 = model(image, cue)
        r3 = F.interpolate(r3, (352, 352), mode='bilinear', )
        r2 = F.interpolate(r2, (352, 352), mode='bilinear', )
        r1 = F.interpolate(r1, (352, 352), mode='bilinear', )
        r0 = F.interpolate(r0, (352, 352), mode='bilinear', )

        # ---- calculate loss ----
        loss3 = structure_loss(mask, r3, kernel_size=31, weights=5.0)
        loss2 = structure_loss(mask, r2, kernel_size=31, weights=5.0)
        loss1 = structure_loss(mask, r1, kernel_size=31, weights=5.0)
        loss0 = structure_loss(mask, r0, kernel_size=31, weights=5.0)
        loss = 0.3 * loss3 + 0.3 * loss2 + 0.2 * loss1 + 0.2 * loss0
        loss = loss.item()

        pred = r3.sigmoid().data.cpu().numpy()
        mask = mask.cpu().numpy()

        pred = 1 * (pred > 0.5)
        mask = 1 * (mask > 0.5)

        dice = mean_dice_np(mask, pred, axes=(2, 3))
        iou = mean_iou_np(mask, pred, axes=(2, 3))
        seg_acc = mean_seg_acc_np(mask, pred)

        dice_bank.append(np.mean(dice))
        iou_bank.append(np.mean(iou))
        loss_bank.append(loss)
        seg_acc_bank.append(seg_acc)

    print('{} Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
          format(val_flag, np.mean(loss_bank), np.mean(dice_bank),
                 np.mean(iou_bank), np.mean(seg_acc_bank)))

    if save_flag and best_dice < np.mean(dice_bank) and epoch >= 15:
        save_path = './save_model/{}/'.format(save_name)
        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), save_path + 'epoch_%d.pth' % epoch)
    return np.mean(dice_bank)


if __name__ == "__main__":
    model = VANet(in_chans=3,
                  act_layer=QuickGELU,
                  cfg='./experiments/imagenet/cvt/cvt-13-384x384.yaml',
                  weights='./weights/CvT-13-384x384-IN-1k.pth',
                  embed_dims=[64, 192, 384],
                  depths=[1, 2, 10],
                  mlp_ratios=[4, 4, 4],
                  num_heads=[1, 3, 6],
                  strides=[4, 2, 2],
                  proj_drop=0.1,
                  attn_drop=0.1,
                  drop_path=0.1,
                  norm_layer=partial(LayerNorm, eps=1e-5),
                  num_class=1, )
    model = model.cuda()
    params = model.parameters()
    # model.load_state_dict(torch.load('/data2/clh/workspace/OANet/save_model/BANet0404/epoch_20.pth'))
    print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    classifier = ResNet50(in_channels=3, num_classes=2, )
    classifier.load_state_dict(torch.load('./weights/ResNet-50-352x352.pth'))
    classifier.cuda()
    classifier.eval()
    cam_extractor = GradCAMpp(classifier, )

    total_epoch = 50
    batch_size = 16
    seed = 205
    grad_norm = 10.0
    base_lr = 2e-4
    save_name = 'VANet_resnet50'
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=1e-4)

    dataset = PolypDataset(shape=(352, 352))
    val_dataset = ValDataset(shape=(352, 352))

    indices = list(range(len(dataset)))
    split = int(np.floor(0.2 * len(dataset)))  # train:val = 8:2
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
    val_best_dice = 0.0
    test_best_dice = 0.0

    print("*" * 20, "|| Start Training ||", "*" * 20)
    for epoch in range(1, total_epoch + 1):
        optimizer.zero_grad()
        train(train_loader,
              model,
              optimizer,
              epoch,
              classifier,
              cam_extractor,
              batch_size=batch_size,
              total_epoch=total_epoch,
              save_name=save_name,
              grad_norm=grad_norm)
        # lr_ = base_lr * (1.0 - epoch / total_epoch) ** 0.9
        # for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr_
        val_dice = val(model,
                       val_best_dice,
                       val_loader,
                       classifier,
                       cam_extractor,
                       epoch=epoch,
                       save_flag=True,
                       save_name=save_name,
                       val_flag='val')
        if val_dice > val_best_dice:
            val_best_dice = val_dice
            print("val best dice is {:.4f}".format(val_dice))
