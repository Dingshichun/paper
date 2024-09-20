'''

'''
import os # 导入操作系统模块
import math # 数学函数
import argparse
import torch
import torch.optim as optim #优化函数
from torch.utils.tensorboard import SummaryWriter #tensorboard可视化工具
from torchvision import transforms #数据增强
import torch.optim.lr_scheduler as lr_scheduler #学习率设置函数
import torchvision

# 接下来的MyDataSet和utils参考劈里啪啦的代码my_dataset.py和utils.py，
# 而model则参考开头注释从efficientnet_pytorch中导入。
from my_dataset import MyDataSet # 从my_dataset导入MyDataSet函数，可返回图像和其标签。
# utils是之前定义过的模块，从中导入一些函数和类。
from utils import read_split_data, train_one_epoch, evaluate # 文件夹标签划分，训练一轮的函数，一个验证函数

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(args) # 打印导入的参数。
    print('Start Tensorboard with "tensorboard --logdir=runs",view at http://localhost:6006/')
    tb_writer = SummaryWriter('D:/FoodScience/Data/MobileNetV2/RunData') # 括号内设置数据可视化路径,默认在当前文件路径创建runs文件夹
    # 该路径用来保存训练的模型
    if os.path.exists("D:/FoodScience/Data/MobileNetV2/ModelSave") is False:
        os.makedirs("D:/FoodScience/Data/MobileNetV2/ModelSave")
    # read_split_data()来自utils模块
    train_images_path,train_images_label,val_images_path,val_images_label \
        = read_split_data(args.data_path)

    # 数据增强方法
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),  # 将原图大小变为299×299
            transforms.RandomRotation(90), # 随机旋转0~90度
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomVerticalFlip(),  # 随机垂直翻转
            transforms.ToTensor(),  # 张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 标准化
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),  # 将原图大小变为299×299
            transforms.ToTensor(),  # 张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
        ])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(
        images_path = train_images_path,
        images_class = train_images_label,
        transform = data_transform["train"]
    )
    # 实例化验证数据集
    val_dataset = MyDataSet(
        images_path = val_images_path,
        images_class = val_images_label,
        transform = data_transform["val"]
    )
    batch_size = args.batch_size
    # nw是加载图像的进程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
    print('Using {} dataloader workers every process'.format(nw))

    # 加载训练数据
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True, #随机打乱
        pin_memory = True,
        # pinned memory，锁页内存，
        # 设置为true时，内存中的tensor转移到GPU上会更快
        num_workers = nw,
        collate_fn = train_dataset.collate_fn  # collate_fn是整理数据的函数。
    )
    # 加载验证数据
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False, # 验证时没必要将数据打乱。
        pin_memory = True,
        num_workers = nw,
        collate_fn = val_dataset.collate_fn
    )
    # 加载模型,不使用博主劈里啪啦的模型，加载自带的经过预训练的模块。
    model = torchvision.models.mobilenet_v2(pretrained=True)
    # print(model)
    model.classifier=torch.nn.Sequential(
        torch.nn.Dropout(p=0.2,inplace=False),
        torch.nn.Linear(in_features=1280,out_features=4,bias=True)
    )
    model = model.to(device)

    if args.freeze_layers: #如果传入的参数freeze_layers为真则表示冻结输出层之外的其它层，只训练输出层的参数。
        for name,para in model.named_parameters():
            # named_parameters()返回各层的参数名称和数据
            if("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    pg = [p for p in model.parameters() if p.requires_grad]
    # 使用带动量的随机梯度下降优化函数
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    for name, para in model.named_parameters():
        # named_parameters()返回各层的参数名称和数据
        if para.requires_grad==True:
            print("training {}".format(name))

    scheduler=lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    val_acc_list = []  # 验证集准确率列表，
    for epoch in range(args.epochs):
        # 每训练一轮的平均损失
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        scheduler.step()  # 改变学习率
        # validate，验证
        # evaluate()函数在博主劈里啪啦自定义的utils模块中。
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        # 将损失loss、准确率accuracy和衰减的学习率learning_rate进行可视化。
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.close()  # 不要忘记关闭

        val_acc_list.append(acc)  # 将每轮验证的准确率装入列表
        max_acc = max(val_acc_list)  # 返回准确率最大值
        max_acc_index = val_acc_list.index(max(val_acc_list))  # 返回最大值索引
        if epoch > 11:
            if val_acc_list[epoch] < max_acc and epoch - max_acc_index  >= 11:
                # 保存模型参数
                torch.save(model.state_dict(),
                           "D:/FoodScience/Data/MobileNetV2/ModelSave/MobileNetV2-{}-state_dict.pth".format(max_acc_index))
                # 保存模型参数和架构
                torch.save(model, "D:/FoodScience/Data/MobileNetV2/ModelSave/MobileNetV2-{}-all.pth".format(max_acc_index))
                break
            else:
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 种类数修改为自己数据集的种类数。
    parser.add_argument('--num_class',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=28)
    parser.add_argument('--lr',type=float,default=0.01)

    # 数据路径
    parser.add_argument('--data_path',type=str,
                        default="E:/HongShu/NewHongShu/HongShu1/TrainVal")

    parser.add_argument('--freeze_layers',type=bool,default=False)
    parser.add_argument('--device',default='cuda:0',help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args(args=[]) # 不传入，使用默认参数。
    main(args) # 调用定义的main()函数。

