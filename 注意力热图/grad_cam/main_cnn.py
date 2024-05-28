import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img

# 改进的模型命名为MB-v3-31-all.pth，未改进的是origin-MB-v3-29-all.pth
# 注意每次修改图片路径img_path时要修改对应的类别target_category
# 类别target_category在imagenet1k_classes.txt文件当中，序号从0开始，
# 发芽FF是0，霉腐MF是1，损伤SS是2，正常ZC是3。
def main():
    # 改进的MBV3,注意要加入改进之后的MBV3原始模型定义文件。
    model = torch.load('D:/PaperDate/Heatmap/model/change_MBV3_model/MB-v3-31-all.pth', map_location=torch.device('cpu'))

    # 未改进的MBV3，注意要加入未改进的MBV3原始模型定义文件。
    # model = torch.load('D:/PaperDate/Heatmap/model/origin_MBV3_model/origin-MB-v3-29-all.pth', map_location = torch.device('cpu'))

    # model = models.mobilenet_v3_large(pretrained=True)
    print(model)
    target_layers = [model.features[-1]]
    # target_layers = [model.features]
    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    # 图片路径待补充完整。
    # 每次修改图片路径之后，甘薯对应的类别也发生变化，
    # 所以后文的target_category也要同时修改，否则效果不是预想。
    img_path = "D:/PaperDate/Heatmap/Test_Picture/FY/FY8.bmp"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # 不使用GPU，使用CPU。
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 0  # txt文件中类别所在的行数，从0开始，第一行是0。

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.axis('off') # 关闭坐标轴
    plt.imshow(visualization)
    plt.show()



if __name__ == '__main__':
    main()
