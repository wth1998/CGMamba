import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from CGDMamba import CGDMamba
import logging

# 配置 logging
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    return train_loss, train_acc


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100.0 * correct / total
    return test_loss, test_acc


def main(train_dir, test_dir, train_epochs, log_file,
         depths=[2, 2, 9, 2], dims=[96, 192, 384, 768]):
    setup_logging(log_file)

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    }

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transform["test"])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 设备设置（使用所有可用的 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型初始化
    model = CGDMamba(depths=depths, dims=dims, num_classes=len(train_dataset.classes)).to(device)
  
    # # 加载预训练权重
    # pretrained_weights = torch.load('/ai/CGDMamba/vssm1_tiny_0230s_ckpt_epoch_264.pth') 
    # load_result = model.load_state_dict(pretrained_weights, strict=False)

    # # 打印加载结果
    # print("Missing keys:", load_result.missing_keys)
    # print("Unexpected keys:", load_result.unexpected_keys)



    # 使用多 GPU 训练
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)  # 使用 DataParallel 包装模型

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)

    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 初始化学习率调度器
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_acc = 0.0

    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch + 1}/{train_epochs}")
        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 测试
        test_loss, test_acc = test(model, test_loader, criterion, device)
        logging.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # # 根据验证损失调整学习率
        # scheduler.step(test_loss)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(), "best_model_dataset_new.pth")
            logging.info(f"Best model saved with accuracy: {best_acc:.2f}%")

    logging.info("Training complete!")


if __name__ == "__main__":
    main(train_dir='/ai/CGDMamba/imagenet1k/train',
         test_dir='/ai/CGDMamba/imagenet1k/val',
         train_epochs=300,
         log_file="training_log_imagenet1k.txt",
        #  depths=[2,2,27,2],
        #  dims=[128,256,512,1024]
         )