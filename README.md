代码来自https://github.com/milesial/Pytorch-UNet

运行方法：

1.```train.py ```训练模型，训练好的模型保存在```/checkpoints```中

2.```predict.py ```预测输出，用```-i```参数指定预测的图像地址

3.```after.py ```对预测的标签做后续处理

4.```eval_score.py``` 计算预测标签的平均纵向误差，并将预测结果叠加回原图

