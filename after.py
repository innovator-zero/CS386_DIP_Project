import os
import cv2
import numpy as np


def bin_blur(img, th):
    h = img.shape[0]
    w = img.shape[1]
    img = cv2.medianBlur(img, 15)  # 中值滤波

    # 阈值化
    for i in range(h):
        for j in range(w):
            if img[i][j] < th:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img


def after_process(pred_path, after_path, k, it):
    pred_file_names = sorted(os.listdir(pred_path))

    for f in pred_file_names:
        pred_name = os.path.splitext(f)
        pred_name = pred_name[0].replace('_OUT', '')  # '_OUT'为文件名后缀

        pred_file = os.path.join(pred_path + f)

        img = cv2.imread(pred_file, 0)

        kernel = np.ones((k, k), np.uint8)
        img = cv2.erode(img, kernel, iterations=it)  # 腐蚀

        img = bin_blur(img, 250)

        cv2.imwrite(os.path.join(after_path + pred_name + '_OUT.jpg'), img)


if __name__ == "__main__":
    pred_path = './output/'  # 预测出来的标签文件夹地址
    after_path = './output_p/'  # 处理后标签存放的文件夹地址

    if not os.path.exists(after_path):
        os.mkdir(after_path)

    after_process(pred_path, after_path, 3, 5)  # kernel size=3，做5次
