# 论文中的主要模型
包括原始的 MobileNetV3Small、MobileNetV3Large、EfficientNetB0，
以及使用 ECA 模块替换 SE 模块之后得到的 ECA_MobileNetV3Small
## （1）**文件简介**
```
├── ECA_MobileNetV3_model.py：        ECA_MobileNetV3 模型  
├── Origin_MobileNetV3_model.py：     原始 MobileNetV3 模型  
├── ECA_MobileNetV3Small_train.py：   训练 ECA_MobileNetV3Small 模型 
├── Origin_MobileNetV3Large_train.py：训练原始 MobileNetV3Large 模型
├── Origin_MobileNetV3Small_train.py：训练原始 MobileNetV3Small 模型
├── test_images：             测试图像
├── class_indices.json：      甘薯类别索引文件    
├── ConfusionMatrix.py：      得到混淆矩阵、模型参数量和计算量         
├── EfficientNet_model.py：   EfficientNet 模型，B0 到 B7               
├── EfficientNetB0_train.py： 训练 EfficientNetB0 模型                 
├── my_dataset.py：           实现自定义数据集       
├── Param-commute.py：        计算模型参数量和计算量的方法
├── requirements.txt：        环境配置
└── utils.py：                划分训练、验证集，实现一轮次的训练，验证模型准确率。   
```
## （2）**使用步骤**
1. 环境配置。提前安装好需要的模块。

2. 下载预训练权重，主要包括 EfficientNetB0 和 MobileNetV3Small 和 MobileNetV3Large 。

3. 运行文件名中含有 train 的几个文件即可训练对应的模型，保存模型。

4. 运行 ConfusionMatrix.py ，加载训练时保存的模型，计算出其参数量和计算量，并绘制出混淆矩阵，得到单张图片的推理时间。