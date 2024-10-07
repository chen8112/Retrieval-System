import numpy as np
import cv2
import os
import csv
import time
import matplotlib.pyplot as plt

def color_histograms(filename,bins):
    img = cv2.imread(filename)  # 读一张彩色图片
    if img is None:
        return
    # 将图像转化为HSV颜色空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR空间转换为HSV空间
    # 计算直方图
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
    # 归一化直方图
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    # 返回一维数组
    return hist.flatten()

def show_colorh(image_path):
    img = cv2.imread(image_path)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 计算H, S, V三个通道的直方图
    h_hist = cv2.calcHist([hsv_img], [0], None, [256], [0, 256])
    s_hist = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])

    # # 绘制直方图
    # plt.figure(figsize=(12, 4))
    #
    # plt.subplot(1, 3, 1)
    # plt.plot(h_hist, color='r')
    # plt.title('H Channel Histogram')
    # plt.xlim([0, 256])
    #
    # plt.subplot(1, 3, 2)
    # plt.plot(s_hist, color='g')
    # plt.title('S Channel Histogram')
    # plt.xlim([0, 256])
    #
    # plt.subplot(1, 3, 3)
    # plt.plot(v_hist, color='b')
    # plt.title('V Channel Histogram')
    # plt.xlim([0, 256])
    #
    # plt.tight_layout()
    # plt.show()
    # 设置绘图
    plt.figure(figsize=(10, 4))

    # 合并直方图到一张图
    bins = np.arange(256)
    plt.bar(bins - 0.3, h_hist.flatten(), width=0.3, color='r', label='H Channel')
    plt.bar(bins, s_hist.flatten(), width=0.3, color='g', label='S Channel')
    plt.bar(bins + 0.3, v_hist.flatten(), width=0.3, color='b', label='V Channel')
    plt.title('Combined H, S, V Channel Histograms')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.legend()

    plt.tight_layout()
    plt.show()

def extract_colorh(image_folder,output_file):
    #获取图像文件列表
    image_files = os.listdir(image_folder)
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        #遍历图像文件列表
        for image_file in image_files:
            if image_file.endswith('.jpg'):
                #读取图像
                image_path = os.path.join(image_folder,image_file)
                #提取特征
                features = color_histograms(image_path,bins=(4,4,4))
                print(image_path,features)
                writer.writerow([image_path] + features.tolist())

    print("特征提取完毕")

# 计算直方图相似度
def compare_histograms(hist1, hist2):
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return similarity

def search_colorh(search_image,output_file):
    features = []
    image_paths = []
    with open(output_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            image_path = row[0]
            feature = np.array([float(val) for val in row[1:]])
            image_paths.append(image_path)
            features.append(feature)

    features = np.array(features)
    search_feature = color_histograms(search_image,bins=(4,4,4)).reshape(1, -1)

    # # 记录检索开始时间
    # start_time = time.time()
    distances = np.linalg.norm(features - search_feature, axis=1)
    # 获取排序后的索引
    sorted_indices = np.argsort(distances)
    # # 记录检索结束时间
    # end_time = time.time()
    # # 计算并打印检索时间
    # search_time = end_time - start_time
    top_n = 10
    return [(image_paths[i], distances[i]) for i in sorted_indices[:top_n]]
    # print(f"检索时间: {search_time:.4f} 秒")
    # # 检索结果
    # for i in range(min(top_n, len(sorted_indices))):
    #     index = sorted_indices[i]
    #     print(f"Match {i + 1}: {image_paths[index]} with distance {distances[index]}")