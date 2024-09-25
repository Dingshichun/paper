import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img


# ECA_MobileNetV3Small模型命名为 MB-v3-31-all.pth ，
# Origin_MobileNetV3Small是 origin-MB-v3-29-all.pth

# 特别注意每次修改图片路径 img_path 时要修改对应的类别 target_category
# 类别 target_category 在 imagenet1k_classes.txt 文件当中，序号从0开始，
# 发芽 FF 是0，霉腐 MF 是1，损伤 SS 是2，正常 ZC 是3。


def main():
    # 改进的 MBV3 ,注意要将改进之后的 MBV3 原始模型定义文件放到文件夹中，这里是 model_v3.py。
    model = torch.load(
        "./model/change_MBV3_model/MB-v3-31-all.pth",
        map_location=torch.device("cpu"),
    )

    # 未改进的 MBV3，注意要将未改进的 MBV3 原始模型定义文件放到文件夹中，这里是 origin_model_v3.py。
    # model = torch.load(
    #     "./model/origin_MBV3_model/origin-MB-v3-29-all.pth",
    #     map_location=torch.device("cpu"),
    # )

    print(model)
    # 这里只看特征提取层中最后一层的注意力热图，即 model.features[-1]
    target_layers = [model.features[-1]]

    data_transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # load image
    # 每次修改图片路径之后，甘薯对应的类别也发生变化，
    # 所以后文的 target_category 也要同时修改，否则不能得到甘薯对应类别的理想热图。
    img_path = "./test_images/FY2.bmp"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert("RGB")
    img = np.array(img, dtype=np.uint8)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # 不使用GPU，使用CPU。
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # 注意每次更改不同类别的图像进行预测都要修改 target_category 为对应的类别
    target_category = 0  # txt文件中类别所在的行数，从0开始，第一行是0。

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(
        img.astype(dtype=np.float32) / 255.0, grayscale_cam, use_rgb=True
    )
    plt.axis("off")  # 关闭坐标轴
    plt.imshow(visualization)
    plt.show()


if __name__ == "__main__":
    main()
