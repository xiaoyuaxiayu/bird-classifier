# Bird Classifier

70 类鸟类图像分类实验代码。数据集不包含在本仓库中。

## 环境

- Python 3.12
- PyTorch
- torchvision
- Pillow

## 训练

将课程数据集放置在以下目录：

```text
实验一数据集/实验一数据集/
├── bird/
│   ├── classes.txt
│   ├── trainDataSet_mixed.txt
│   └── images/
└── test_DataSet/
```

运行训练：

```powershell
python bird_classifier.py train --epochs 10 --fine-tune-last-block --learning-rate 0.0001
```

训练过程会输出每个 Epoch 的 Loss、训练准确率和验证准确率，并保存：

- `outputs/best.pt`：验证集表现最佳的检查点
- `outputs/last.pt`：最后一轮、可继续训练的检查点

## 生成提交文件

```powershell
python bird_classifier.py predict `
  --checkpoint outputs\best.pt `
  --student-id 你的学号 `
  --student-name 你的姓名 `
  --output outputs
```

程序会对 1666 张测试图片进行预测，并输出符合课程要求的文本文件。测试集没有公开标签，因此程序不会虚构测试集准确率。
