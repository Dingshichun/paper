import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model_v3 import mobilenet_v3_small
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter('D:/FoodScience/Data/MobileNetV3/RunData')
    # tensorboard文件保存的路径
    if os.path.exists("D:/FoodScience/Data/MobileNetV3/RunData") is False:
        os.makedirs("D:/FoodScience/Data/MobileNetV3/RunData")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = {
        "train": transforms.Compose([transforms.Resize(img_size[num_model]),
                                     transforms.RandomRotation(90),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False, # 验证集的数据不用打乱
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入，create_model是EfficientNet-b0的别名。
    model = mobilenet_v3_small(num_classes=args.num_classes).to(device)

    print(model)
    # if args.weights != "":
    #     if os.path.exists(args.weights):
    #         weights_dict = torch.load(args.weights, map_location=device)
    #         load_weights_dict = {k: v for k, v in weights_dict.items()
    #                              if model.state_dict()[k].numel() == v.numel()}
    #         print(model.load_state_dict(load_weights_dict, strict=False))
    #     else:
    #         raise FileNotFoundError("not found weights file: {}".format(args.weights))
    '''
    # # 对模型做了修改之后加载未修改部分的预训练权重
    # ckpt = torch.load(weights)  # 加载预训练权重
    # model_dict = model.state_dict()  # 得到我们模型的参数
    # # 判断预训练模型中网络的模块是否修改后的网络中也存在，并且shape相同，如果相同则取出
    # pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and (v.shape == model_dict[k].shape)}
    # # 更新修改之后的 model_dict
    # model_dict.update(pretrained_dict)
    # # 加载我们真正需要的 state_dict
    # model.load_state_dict(model_dict, strict=False)
    '''

    MB_V3_pretrained_dict=torch.load(args.weights,map_location=device)
    model_dict=model.state_dict()
    pretrained_dict={k: v for k, v in MB_V3_pretrained_dict.items() if k in model_dict and (v.shape == model_dict[k].shape)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    # 是否冻结权重，默认为False。
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    for name,para in model.named_parameters():
        if para.requires_grad==True:
            print("training {}".format(name))
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    scheduler=lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    val_acc_list = []  # 验证集准确率列表，
    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.close()  # 不要忘记关闭

        # 设置早停模式，验证集的准确率连续15轮低于最高准确率则退出，同时保存最高准确率的模型。
        val_acc_list.append(acc)  # 将每轮验证的准确率装入列表
        max_acc = max(val_acc_list)  # 返回准确率最大值
        max_acc_index = val_acc_list.index(max(val_acc_list))  # 返回最大值索引
        if epoch > 11:
            if val_acc_list[epoch] < max_acc and epoch - max_acc_index >= 11:
                # 这样保存的好像只是当前训练批次的模型，只是命名的时候加上了最高准确率的位置？
                # 保存验证准确率最高的模型参数
                torch.save(model.state_dict(),
                           "D:/FoodScience/Data/MobileNetV3/ModelSave/MB-v3-{}-state_dict.pth".format(
                               max_acc_index))
                # 保存证准确率最高的模型参数和架构
                torch.save(model,
                           "D:/FoodScience/Data/MobileNetV3/ModelSave/MB-v3-{}-all.pth".format(
                               max_acc_index))
                break
            else:
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=28)
    parser.add_argument('--lr', type=float, default=0.01)


    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="E:/HongShu/NewHongShu/HongShu1/TrainVal")

    # download model weights
    # 链接: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090i
    parser.add_argument('--weights', type=str, default='./mobilenet_v3_small.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args(args=[])  # 不传入，使用默认参数。
    main(args)  # 调用定义的main()函数。

