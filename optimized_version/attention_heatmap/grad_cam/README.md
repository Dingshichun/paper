## Grad-CAM
- Original Impl: [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

- Grad-CAM简介: [https://b23.tv/1kccjmb](https://b23.tv/1kccjmb)

- 使用Pytorch实现Grad-CAM并绘制热力图: [https://b23.tv/n1e60vN](https://b23.tv/n1e60vN)

## （1）**文件简介**
```
├── model:                  ECA_MobileNetV3Small 和 Origin_MobileNetV3Small
├── utils.py：              实现 Grad_CAM 主要功能
├── test_images:            甘薯测试图
├── main_cnn.py：           得到注意力热图的主程序
├── model_v3.py：           改进的 MobileNetV3Small 模型实现
├── origin_model_v3.py：    原始 MobileNetV3Small 模型实现
└── imagenet1k_classes.txt: 甘薯类别  
```
## （2）**使用步骤**(替换成自己的网络)
1. 将创建模型部分代码替换成自己创建模型的代码，并载入自己训练好的权重。具体就是将 `model_v3.py` 和 `origin_model_v3.py` 替换成自己要生成的模型，然后再将 `model` 文件夹中的模型换成自己训练好的模型。

2. 根据自己网络设置合适的`target_layers`。即更改 `imagenet1k_classes.txt` 文件中的类别，第一个类别对应 id 序号 0 ，第二个对应 1 ，依次类推。

3. 根据自己的网络设置合适的预处理方法

4. 将要预测的图片路径赋值给 `img_path`

5. 将要预测的类别 id 值赋给 `target_category`

