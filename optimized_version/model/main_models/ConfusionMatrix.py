"""
利用测试数据生成混淆矩阵
"""

import os
import json  # （JavaScript Object Notation）
import time  # 用于记录程序运行时间
import torchvision
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm  # tqdm显示进度
import matplotlib.pyplot as plt
from prettytable import PrettyTable  # 打印表格样式的模块
from thop import profile


# 定义一个混淆矩阵类
class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(
        self, num_classes: int, labels: list
    ):  # num_classes是类别数，labels则传入标签列表
        self.matrix = np.zeros((num_classes, num_classes))  # 用0初始化混淆矩阵
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):  # 把预测值和真实值传入
        for p, t in zip(preds, labels):  # p表示预测值，t表示真实值，zip是打包压缩
            self.matrix[
                p, t
            ] += 1  # 在第p行第t列累加，现在每行表示预测的值，每列表示真实的值

    def summary(self):  # 显示各项指标的函数
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵的对角线表示TP
        acc = sum_TP / np.sum(self.matrix)  # 计算准确度
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()  # 生成一个表
        table.field_names = ["", "Precision", "Recall", "Specificity"]  # 表格的第一行
        for i in range(self.num_classes):
            TP = self.matrix[i, i]  # 对照混淆矩阵可得各项数据
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = (
                round(TP / (TP + FP), 4) if TP + FP != 0 else 0.0
            )  # round表示小数部分取三位
            Recall = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.0
            Specificity = round(TN / (TN + FP), 4) if TN + FP != 0 else 0.0
            table.add_row([self.labels[i], Precision, Recall, Specificity])
            # 将各个类别的各项指标加入到表格
        print(table)

    def plot(self):
        matrix = self.matrix  # 复制混淆矩阵
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)  # cmap参数修改混淆矩阵颜色。
        # 颜色格式包括hot、jet、grey、bone和hsv等。
        # 设置x轴坐标label，如果标签太长导致水平显示时不完整，可将其旋转45度。
        plt.xticks(range(self.num_classes), self.labels, rotation=0)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()  # 打开颜色条
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.title("Confusion matrix")

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2  # 设置一个阈值来决定颜色深浅
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]。混淆矩阵y轴从上到下逐渐变大
                info = int(matrix[y, x])
                plt.text(
                    x,
                    y,
                    info,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="white" if info > thresh else "black",
                )  # 根据阈值设置颜色
        plt.tight_layout()  # 使图框更加紧凑
        plt.show()


if __name__ == "__main__":
    # 使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 数据预处理方式应该和训练模型一致
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_path = "E:/HongShu/NewHongShu/HongShu1/Test"  # 测试数据路径
    test_set = torchvision.datasets.ImageFolder(
        root=test_path, transform=data_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=28, shuffle=False  # 加载测试数据
    )

    # 加载训练之后保存的模型用于测试
    model = torch.load("D:/FoodScience/Data/ResNet50/ModelSave/ResNet50-50-all.pth")
    model.to(device)
    # read class_indict
    json_label_path = "class_indices.json"  # 这是一个以字典形式存储类别标签的文件
    assert os.path.exists(json_label_path), "cannot find {} file".format(
        json_label_path
    )
    json_file = open(json_label_path, "r")
    class_indict = json.load(json_file)

    """
    上面存储标签的文件应该可以直接生成一个字典，使用字典即可
    """
    labels = [
        label for _, label in class_indict.items()
    ]  # 利用循环得到字典中存储的标签值
    confusion = ConfusionMatrix(num_classes=4, labels=labels)  # 实例化混淆矩阵
    model.eval()  # 切换到神经网络的验证模式。预训练的网络一般都有train和val模式

    with torch.no_grad():  # pytorch不再计算梯度，减小显存
        for test_data in tqdm(test_loader):  # tqdm显示加载进度。
            test_images, test_labels = test_data
            outputs = model(test_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())

    confusion.plot()
    confusion.summary()

    # 使用thop模块计算模型参数量和计算量FLOPs
    # 注意，如果是使用GPU训练得到的模型，这里用于计算参数量和计算量的dummy_input也要移动到gpu上
    # 否则CPU和GPU上得到的结果可能不一样。
    dummy_input = torch.randn(28, 3, 224, 224, device=device)
    FLOPs, params = profile(model, (dummy_input,))
    print("FLOPs: ", FLOPs, "params: ", params)
    print("FLOPs: %.2f M, params: %.2f M" % (FLOPs / 1000000.0, params / 1000000.0))
    # 加载一幅图用于计算推理时间
    from PIL import Image

    img = Image.open("SunShang.bmp")

    # 计算模型推理时间
    torch.cuda.synchronize()
    start = time.perf_counter()

    # 由于某些要求的输入图像尺寸还包括batchsize，且是第一个维度，所以增加一个维度。
    img_ = data_transform(img).unsqueeze(0)
    result = model(img_.to(device))
    result = torch.softmax(result, dim=1)
    result = torch.argmax(result, dim=1)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(result)
    print("处理速度:", 1 / (end - start), " 幅/秒")
