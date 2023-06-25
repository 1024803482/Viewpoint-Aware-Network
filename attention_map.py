from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import GradCAMpp, SmoothGradCAMpp
from torchcam.utils import overlay_mask
from Classifier import ResNet50
import torch
import imageio
import os
import cv2


def display_map(image_path, cam_extractor):
    img = read_image(image_path)
    input_tensor = normalize(resize(img, (352, 352)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    out = model(input_tensor.unsqueeze(0))
    activation_map = cam_extractor(1, out)
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    return activation_map[0].squeeze(0).numpy(), result


def get_map(image_path, cam_extractor):
    img = read_image(image_path)
    input_tensor = normalize(resize(img, (352, 352)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    out = model(input_tensor.unsqueeze(0))
    activation_map = cam_extractor(1, out)
    return activation_map[0].squeeze(0).numpy()


if __name__ == "__main__":
    model = ResNet50(in_channels=3, num_classes=2, )
    model.load_state_dict(torch.load('./weights/ResNet-50-352x352.pth'))
    model.eval()
    cam_extractor = GradCAMpp(model)
    # Get your input
    root_path = "./Datasets/VANet_Dataset/TestDataset/CVC-ClinicDB/"
    save_path = "./visualization/ResNet50_Gradpp/CVC-ClinicDB/"

    for label in os.listdir(root_path):
        if label == 'mask':
            continue
        dir_path = root_path + label + '/'
        os.makedirs(save_path + label, exist_ok=True)
        for name in os.listdir(dir_path):
            print(name)
            if name[0] == '.':
                continue
            image_path = dir_path + name
            cam, im = display_map(image_path, cam_extractor)

            imageio.imwrite(save_path + label + '/' + name, im)
            cam = cv2.resize(cam, (330, 330), interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(save_path + label + '/' + name.split('.')[0] + '_cam.png', cam)