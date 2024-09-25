# %%%%%%%%%%%%%%%%
# "# %%%"可以直接生成可运行的单元格。

# 计算模型参数量和计算量，使用thop模块
import math
import torchvision
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 注意之前的模型使用的是gpu进行训练，得到的也是gpu类型的模型
# 所以后边输入的随机张量也要转移到gpu上才行。
model = torch.load(
    "E:/PaperModels/MobileNetV2/MobileNetV2_Model/MobileNetV2-10-all.pth"
)
import thop
from thop import profile

model.to(device)


dummy_input = torch.randn(28, 3, 224, 224, device=device)
flops, params = profile(model, (dummy_input,))
print("flops: ", flops, "params: ", params)
print("flops: %.2f M, params: %.2f M" % (flops / 1000000.0, params / 1000000.0))

# 输入input的第一维度是批量(batch size)，批量的大小不会影响参数量，但是影响计算量，计算量是batch_size=1的倍数
# profile(net, (inputs,))的 (inputs,)中必须加上逗号，否者会报错
# %%%%%%%%%%%%%%%%%%%%%%%%%
