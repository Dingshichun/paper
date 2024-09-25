# 论文中用于对比的模型
包括 AlexNet、GoogLeNet、MobileNetV2、ResNet50、ResNet101、ShuffleNet、SqueezeNet、VGG19 。
## （1）**文件简介**
```
├── class_indices.json：  保存类别索引                
├── ConfusionMatrix.py：  得到混淆矩阵、模型参数量和计算量
├── my_dataset.py：       实现自定义数据集       
├── requirements.txt：    环境配置
├── utils.py：            划分训练、验证集，实现一轮次的训练，验证模型准确率。   
└── 其余文件：             其它文件则是训练其文件名对应模型的脚本
```
## （2）**使用步骤**
1. 环境配置。提前安装好需要的模块。

2. 运行文件，训练对应的模型，保存模型。

3. 运行 ConfusionMatrix.py ，加载训练时保存的模型，计算出其参数量和计算量，并绘制出混淆矩阵，得到单张图片的推理时间。