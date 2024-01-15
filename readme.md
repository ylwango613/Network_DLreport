# 实验环境
安装好conda后在终端输入以下命令即可：
```
conda create -n dl python=3.10.8
conda activate dl
git clone https://github.com/ylwango613/Network_DLreport
cd Network_DLreport
pip install -r requirements.txt
```

# 数据集下载
使用的是cifar10数据集，已经内置在了torchvision中，代码中为：

```
# 加载和预处理CIFAR-10数据集
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

```
这里为了保证运行的效果，并未直接对数据集在load阶段进行归一化，有需要可以根据自己面临的任务自行进行归一化处理。

# 运行方式
均给出具体的ipynb文件了，只需要按照jupyter noterbook的方式运行即可。
# 实验结果
|                模型                 | 准确度 |
| :---------------------------------: | :----: |
|      原始图像加强V2LeNet5大核       | 62.55% |
|      原始图像加强V1LeNet5大核       | 63.43% |
|             大核LeNet-5             | 63.52% |
|             原始LeNet5              | 66.09% |
|        原始图像加强V1LeNet5         | 68.34% |
|        原始图像加强V2LeNet5         | 69.03% |
| LeNet5大核卷积提取，单层transformer | 69.55% |
| LeNet5大核卷积提取，双层transformer | 70.43% |
|             原始单层ViT             | 73.28% |
|   LeNet5卷积提取，单层transformer   | 77.43% |
|   LeNet5卷积提取，双层transformer   | 78.25% |
|             原始双层ViT             | 78.48% |
|            原始ResNet18             | 86.38% |
|  ResNet18卷积提取，双层transformer  | 87.47% |
|  ResNet18卷积提取，单层transformer  | 88.27% |