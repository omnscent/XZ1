# XZ1
数字图像处理选做作业1代码

基本的使用方法都和大作业中的代码类似，只需要通过选择对应部分的代码就可以运行不同类型的模型。

数据集的选择有`MNIST_dataset`,`FashionMNIST_dataset`和`CIFAR_dataset`，如果使用`CIFAR_dataset`需要将`input_chann_num`修改成3，其他两个都是1。对于前三个方法，都默认使用`Pi_model`，而对于最后一个方法，需要将模型改变成`cifar_shakeshake26`。

Cleanlab的例子放在了`Cleanlab.py`中。
