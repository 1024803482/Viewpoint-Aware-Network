import torch
import numpy as np
from datetime import datetime
import imageio
import os
import xlwt
import torch.nn.functional as F

from VANet import VANet
from utils import structure_loss, mean_dice_np, mean_iou_np, mean_seg_acc_np, AvgMeter
from PolypDataset import TestDataset
from lib.models import QuickGELU, LayerNorm
from functools import partial
from torchcam.methods import GradCAMpp, SmoothGradCAMpp
from Classifier import ResNet50


def test(model,
         val_loader,
         classifier,
         cam_extractor,
         visualization=False,
         save_path="./visualization/model/", ):
    model.eval()
    name_bank = []
    dice_bank = []
    iou_bank = []
    label_bank = []
    loss_bank = []
    seg_acc_bank = []

    parallel_dice_list = []
    vertical_dice_list = []
    parallel_iou_list = []
    vertical_iou_list = []

    for step, pack in enumerate(val_loader, start=1):
        # ---- data prepare ----
        image, label, mask, name = pack
        image = image.cuda()
        mask = mask.cuda()
        name = name[0].split('/')[-1]
        name_bank.append(name)
        cue = None
        out = classifier(image)
        none_map = cam_extractor(1, out)[0]  # b h w
        if torch.sum(torch.isnan(none_map)) > 0:
            print("nan exists")
        else:
            cue = none_map.unsqueeze(1)

        with torch.no_grad():
            r4, r3, r2, r1 = model(image, cue)
        r4 = F.interpolate(r4, (352, 352), mode='bilinear', )
        r3 = F.interpolate(r3, (352, 352), mode='bilinear', )
        r2 = F.interpolate(r2, (352, 352), mode='bilinear', )
        r1 = F.interpolate(r1, (352, 352), mode='bilinear', )
        # ---- calculate loss ----
        loss = structure_loss(mask, r4, kernel_size=31, weights=5.0)
        loss = loss.item()

        pred4 = r4.sigmoid().data.cpu().numpy()
        pred3 = r3.sigmoid().data.cpu().numpy()
        pred2 = r2.sigmoid().data.cpu().numpy()
        pred1 = r1.sigmoid().data.cpu().numpy()
        mask = mask.cpu().numpy()
        cue = cue.cpu().numpy()

        pred4 = 1 * (pred4 > 0.5)
        pred3 = 1 * (pred3 > 0.5)
        pred2 = 1 * (pred2 > 0.5)
        pred1 = 1 * (pred1 > 0.5)
        mask = 1 * (mask > 0.5)

        if visualization is True:
            imageio.imwrite(save_path + '4/' + name, 255 * np.uint8(pred4.squeeze()))
            imageio.imwrite(save_path + '3/' + name, 255 * np.uint8(pred3.squeeze()))
            imageio.imwrite(save_path + '2/' + name, 255 * np.uint8(pred2.squeeze()))
            imageio.imwrite(save_path + '1/' + name, 255 * np.uint8(pred1.squeeze()))
            imageio.imwrite(save_path + 'cue/' + name.split('.')[0] + '_cue.png', 255 * np.uint8(cue.squeeze()))
        dice = mean_dice_np(mask, pred1, axes=(2, 3))
        iou = mean_iou_np(mask, pred1, axes=(2, 3))
        seg_acc = mean_seg_acc_np(mask, pred4)

        dice_bank.append(dice)
        iou_bank.append(iou)
        loss_bank.append(loss)
        seg_acc_bank.append(seg_acc)
        label_bank.append(label)

        if label == 0:
            parallel_dice_list.append(dice)
            parallel_iou_list.append(iou)
        elif label == 1:
            vertical_dice_list.append(dice)
            vertical_iou_list.append(iou)

    print('{} Loss: {:.4f}, Dice: {:.4f}±{:.4f}, IoU: {:.4f}±{:.4f}, Acc: {:.4f}±{:.4f}'.
          format('test', np.mean(loss_bank), np.mean(dice_bank), np.std(dice_bank),
                 np.mean(iou_bank), np.std(iou_bank), np.mean(seg_acc_bank), np.std(seg_acc_bank)))
    print('{} Dice: {:.4f}±{:.4f}, IoU: {:.4f}±{:.4f}'.
          format('parallel', np.mean(parallel_dice_list), np.std(parallel_dice_list),
                 np.mean(parallel_iou_list), np.std(parallel_iou_list), ))
    print('{} Dice: {:.4f}±{:.4f}, IoU: {:.4f}±{:.4f}'.
          format('vertical', np.mean(vertical_dice_list), np.std(vertical_dice_list),
                 np.mean(vertical_iou_list), np.std(vertical_iou_list), ))

    return name_bank, label_bank, loss_bank, dice_bank, iou_bank, seg_acc_bank


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
    test_path = "./Datasets/VANet_Dataset/TestDataset/"
    weight_path = "./save_model/VANet/epoch_50.pth"
    save_path = "./visualization/VANet/"
    # record_path = "./visualization/temp/results.xls"
    visualization = True

    model.load_state_dict(torch.load(weight_path), strict=False)
    model = model.cuda()

    classifier = ResNet50(in_channels=3, num_classes=2, )
    classifier.load_state_dict(
        torch.load('./weights/ResNet-50-352x352.pth'))
    classifier.cuda()
    classifier.eval()
    cam_extractor = GradCAMpp(classifier, )

    print("*" * 20, "|| Start Testing ||", "*" * 20)
    step = 1
    for dir_name in os.listdir(test_path):
        if visualization is True:
            os.makedirs(save_path + dir_name + '/4', exist_ok=True)
            os.makedirs(save_path + dir_name + '/3', exist_ok=True)
            os.makedirs(save_path + dir_name + '/2', exist_ok=True)
            os.makedirs(save_path + dir_name + '/1', exist_ok=True)
            os.makedirs(save_path + dir_name + '/cue', exist_ok=True)
        print("{} Test Result:".format(dir_name))
        test_dataset = TestDataset(root_path=test_path + dir_name + '/',
                                   shape=(352, 352))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)
        name_list, label_list, loss_list, dice_list, iou_list, seg_acc_list = test(model, test_loader,
                                                                                   classifier, cam_extractor,
                                                                                   visualization=visualization,
                                                                                   save_path=save_path + dir_name + '/')
