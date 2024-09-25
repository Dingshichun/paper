"""
训练 ResNet101
"""

import os  # 导入操作系统模块
import argparse
import torch
import torch.optim as optim  # 优化函数
from torch.utils.tensorboard import SummaryWriter  # tensorboard可视化工具
from torchvision import transforms  # 数据增强
import torch.optim.lr_scheduler as lr_scheduler  # 学习率设置函数
import torchvision

# 接下来的MyDataSet和utils参考劈里啪啦的代码my_dataset.py和utils.py，
# 而model则参考开头注释从efficientnet_pytorch中导入。
from my_dataset import MyDataSet  # 从my_dataset导入MyDataSet函数，可返回图像和其标签。

# utils是定义的模块，从中导入一些函数和类。
from utils import (
    read_split_data,
    train_one_epoch,
    evaluate,
)  # 文件夹标签划分，训练一轮的函数，一个验证函数


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)  # 打印导入的参数。
    print(
        'Start Tensorboard with "tensorboard --logdir=runs",view at http://localhost:6006/'
    )

    # 该路径用来保存训练的模型
    if os.path.exists(args.RunData_save_path) is False:
        os.makedirs(args.RunData_save_path)
    # 括号内设置数据可视化路径,默认在当前文件路径创建runs文件夹
    tb_writer = SummaryWriter(args.RunData_save_path)

    # read_split_data()来自utils模块
    train_images_path, train_images_label, val_images_path, val_images_label = (
        read_split_data(args.data_path)
    )

    # 数据增强方法
    data_transform = {
        "train": transforms.Compose(
            [
                # 输入的图像大小为224×224,只传入一个参数时默认该参数为图像短边，长边会同比例放大。
                # 所以需要传入一个元组（224,224），将其长短边都进行指定才能得到想要的效果。
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomRotation(90),  # 随机旋转0~90度
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.RandomVerticalFlip(),  # 随机垂直翻转
                transforms.ToTensor(),  # 张量
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # 标准化
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"],
    )

    # 实例化验证数据集
    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"],
    )

    # nw是加载图像的进程数
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 4])
    print("Using {} dataloader workers every process".format(nw))

    # 加载训练数据
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 随机打乱
        pin_memory=True,
        # pinned memory，锁页内存，
        # 设置为true时，内存中的tensor转移到GPU上会更快
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,  # collate_fn是整理数据的函数。
    )
    # 加载验证数据
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 验证时没必要将数据打乱。
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )
    # 加载官方预训练模型。
    model = torchvision.models.resnet101(weights=True)
    # print(model)# 查看模型以更改最后的输出种类
    feature = model.fc.in_features
    # in_features是该模型最后一个全连接层的输入特征大小
    # 更改最后的输出种类数以满足自己的数据
    model.fc = torch.nn.Linear(
        in_features=feature, out_features=args.num_class, bias=True
    )
    model = model.to(device)

    # 是否冻结权重，默认为False。
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除了全连接层，其他权重全部冻结
            if "fc" in name:
                para.requires_grad_(True)
            else:
                para.requires_grad_(False)

    # 这里是要求梯度更新的参数，用来输入优化器进行更新。
    pg = [p for p in model.parameters() if p.requires_grad]
    # 使用带动量的随机梯度下降优化函数
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    for name, para in model.named_parameters():
        # named_parameters()返回各层的参数名称和数据
        if para.requires_grad:
            print("training {}".format(name))

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    val_acc_list = []  # 存储验证准确率
    for epoch in range(args.epochs):
        # 每训练一轮的平均损失
        mean_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )
        scheduler.step()  # 改变学习率
        # validate，验证
        # evaluate()函数在博主劈里啪啦自定义的utils模块中。
        acc = evaluate(model=model, data_loader=val_loader, device=device)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        # 将损失loss、准确率accuracy和衰减的学习率learning_rate进行可视化。
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.close()  # 不要忘记关闭

        # 这里的想法是先保存第 11 个训练轮次之后的所有模型，因为要保证达到最低训练轮次11.
        # 再根据后面所得验证集准确率最高的模型索引，
        # 即可知晓哪个 epoch 训练的模型验证准确率最高，
        # 最后在保存的模型中选取对应的模型，删掉其它多余的模型即可。

        # 保存全部轮次训练的模型而不是仅保存验证准确率最高的模型，
        # 是因为训练到后面的轮次模型参数会更新，
        # 在当前训练轮次无法回溯去保存验证准确率最高的轮次所对应的模型。

        if epoch > 11:
            # 保存第 11 轮次之后训练的所有模型参数
            torch.save(
                model.state_dict(),
                args.model_save_path + "{}-state_dict.pth".format(max_acc_index),
            )
            # 保存第 11 轮次之后训练的所有模型参数和架构
            torch.save(
                model,
                args.model_save_path + "{}-all.pth".format(max_acc_index),
            )

        val_acc_list.append(acc)  # 将每轮验证的准确率装入列表
        max_acc = max(val_acc_list)  # 返回准确率最大值
        max_acc_index = val_acc_list.index(max(val_acc_list))  # 返回最大值索引

        # 设置早停模式，验证集的准确率连续 11 轮低于最高准确率则退出。
        if epoch > 11:
            if val_acc_list[epoch] < max_acc and epoch - max_acc_index >= 11:
                break
            else:
                continue
    print("验证准确率最高的模型对应的训练轮次是：{}".format(max_acc_index))
    print("最高验证准确率为：{}".format(max_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 种类数修改为自己数据集的种类数。
    parser.add_argument("--num_class", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--freeze_layers", type=bool, default=True)
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )
    parser.add_argument(
        "--img-size", type=int, default=224, help="the size of input image"
    )  # 输入图像的尺寸

    # 数据集所在根目录
    parser.add_argument(
        "--data-path", type=str, default="E:/HongShu/NewHongShu/HongShu1/TrainVal"
    )

    # 保存训练时得到的准确率、损失等数据的 RunData 文件夹路径。
    parser.add_argument(
        "--RunData-save-path",
        type=str,
        default="./Data/ResNet101/RunData",
        help="path of save RunData file",
    )

    # 用来保存模型的路径
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="./Data/ResNet101/model/ResNet101-",
        help="path of save model",
    )
    args = parser.parse_args(args=[])  # 不传入，使用默认参数。
    main(args)  # 调用定义的main()函数。
