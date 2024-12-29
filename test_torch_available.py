import torch

print(torch.__version__)  # 查看torch当前版本号
print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号
print(torch.cuda.is_available())  # 查看当前cuda是否可用于当前版本的Torch，如果输出True，则表示可用
print(torch.backends.cudnn.version()) # 查看cudnn版本是否正确
print(torch.backends.cudnn.is_available())
# 导入torch中的 adam 优化器
