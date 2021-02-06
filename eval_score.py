import cv2
import os
import numpy as np


def mean_yaxis_error_mat(pred, label):
    # pred and label should all in the shape of img_shape
    assert pred.shape == label.shape

    img_shape = pred.shape
    y_errors = np.zeros(img_shape[1], dtype=np.float32)
    y_ranges = [i for i in range(img_shape[0])]
    y_ranges = np.array(y_ranges)

    for i in range(img_shape[1]):
        if (label[:, i] == 0).all() or (pred[:, i] == 0).all():
            continue
        label_y = np.sum(label[:, i] * y_ranges) / np.sum(label[:, i])
        pred_y = np.sum(pred[:, i] * y_ranges) / np.sum(pred[:, i])
        y_errors[i] = np.absolute(label_y - pred_y)

    return np.mean(y_errors)


def eval_score(pred_path, label_path, pred_suffix, label_suffix, output=False, output_path=None, img_path=None):
    pred_file_names = sorted(os.listdir(pred_path))
    label_file_names = sorted(os.listdir(label_path))

    assert len(pred_file_names) == len(label_file_names)

    nums = len(pred_file_names)
    all = 0

    for i in range(nums):
        # 将文件名去掉后缀，以便匹配
        pred_name = os.path.splitext(pred_file_names[i])
        pred_name = pred_name[0].replace(pred_suffix, '')

        label_name = os.path.splitext(label_file_names[i])
        label_name = label_name[0].replace(label_suffix, '')

        if pred_name != label_name:
            print('Wrong match:', pred_file_names[i], label_file_names[i])
            exit()

        print(pred_name)

        pred_file = os.path.join(pred_path + pred_file_names[i])
        label_file = os.path.join(label_path + label_file_names[i])
        img_file = os.path.join(img_path + pred_name + '.tif')

        pred = cv2.imread(pred_file, 0)
        label = cv2.imread(label_file, 0)
        img = cv2.imread(img_file)

        # 计算误差
        all += mean_yaxis_error_mat(pred, label)

        # 将预测结果叠加在原图上
        if output:
            if output_path == None or img_path == None:
                print('No output path or image path!')
                return
            else:
                if not os.path.exists(output_path):
                    os.mkdir(output_path)

                for h_x in range(pred.shape[0]):
                    for w_x in range(pred.shape[1]):
                        if pred[h_x][w_x] >= 250:
                            img[h_x][w_x] = [0, 0, 255]  # 预测结果以红色表示

                cv2.imwrite(output_path + pred_name + '.jpg', img)

    print('Mean error: %.4f' % (all / nums))


if __name__ == "__main__":
    pred_path = './output3_p/'  # 预测出来的标签文件夹地址
    pred_suffix = '_OUT'  # 预测结果的文件名后缀
    label_path = './data/train_label2/'  # 真实标签文件夹地址
    label_suffix = '_mask2'  # 真实标签文件名后缀
    img_path = './data/train_img/'  # 原始图片文件夹地址
    output_path = './res2/'  # 叠加回原图后输出文件夹地址
    eval_score(pred_path, label_path, pred_suffix, label_suffix, True, output_path, img_path)
