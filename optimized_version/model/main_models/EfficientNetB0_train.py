"""
EfficientNetB0 训练代码
"""

import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler  # 学习率自动更新

from EfficientNet_model import efficientnet_b0
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print(
        'Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/'
    )

    if os.path.exists(args.RunData_save_path) is False:
        os.makedirs(args.RunData_save_path)
    # tensorboard文件保存的路径
    tb_writer = SummaryWriter(args.RunData_save_path)

    train_images_path, train_images_label, val_images_path, val_images_label = (
        read_split_data(args.data_path)
    )

    data_transform = {
        "train": transforms.Compose(
            [
                # 传入一个元组，只传入一个数只能使短边达到要求。
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

    # number of workers
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 4])
    print("Using {} dataloader workers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 验证集的数据不用打乱
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    # 实例化模型
    model = efficientnet_b0(num_classes=args.num_classes).to(device)

    # 如果存在预训练权重则将其权重载入模型，减少模型训练时间。
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {
                k: v
                for k, v in weights_dict.items()
                if model.state_dict()[k].numel() == v.numel()
            }
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重，默认为False。
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除了全连接层，其他权重全部冻结
            if "classifier" in name:
                para.requires_grad_(True)
            else:
                para.requires_grad_(False)

    # 这里是要求梯度更新的参数，用来输入优化器进行更新。
    pg = [p for p in model.parameters() if p.requires_grad]
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("training {}".format(name))

    # 冻结一些层的参数后，其参数不再更新，所以不再传入优化器。
    # 只将没有冻结的层的参数传入优化器，节省时间。
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    val_acc_list = []  # 存储验证准确率，

    for epoch in range(args.epochs):
        # train_one_epoch 是 utils.py 中实现一轮训练的函数，
        # 其中已经包含了参数更新和梯度置零，所以这里只使用 scheduler.step() 更新学习率。
        mean_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )

        scheduler.step()

        # validate
        acc = evaluate(model=model, data_loader=val_loader, device=device)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.close()  # 不要忘记关闭

        # 这里的想法是先保存第 11 个训练轮次之后的所有模型，因为要保证达到最低训练轮次11.
        # 再根据后面所得验证集准确率最高的模型索引，
        # 即可知晓哪个epoch训练的模型验证准确率最高，
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
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--lr", type=float, default=0.01)

    # 自行下载预训练权重
    parser.add_argument(
        "--weights",
        type=str,
        default="./efficientnetb0.pth",
        help="initial weights path",
    )
    parser.add_argument("--freeze-layers", type=bool, default=False)
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )

    # 数据集所在根目录，根据实际设置。
    parser.add_argument(
        "--data-path", type=str, default="E:/HongShu/NewHongShu/HongShu1/TrainVal"
    )

    # 保存训练时得到的准确率、损失等数据的 RunData 文件夹路径。
    parser.add_argument(
        "--RunData-save-path",
        type=str,
        default="./Data/EfficientNetB0/RunData",
        help="path of save RunData file",
    )

    # 用来保存模型的路径
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="./Data/EfficientNetB0/model/EfficientNetB0-",
        help="path of save model",
    )
    args = parser.parse_args(args=[])  # 不传入，使用默认参数。
    main(args)  # 调用定义的main()函数。
