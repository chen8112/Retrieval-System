import numpy as np
import cv2
import os
import csv
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from skimage.feature import graycomatrix, graycoprops

def color_moments(filename):
    # 颜色矩
    img = cv2.imread(filename)  # 读一张彩色图片
    if img is None:
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR空间转换为HSV空间
    h, s, v = cv2.split(hsv)
    color_feature = []  # 初始化颜色特征
    # 计算每个通道的颜色矩，一阶矩（均值 mean）
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])  # 一阶矩放入特征数组
    # 二阶矩 （标准差 std）
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])  # 二阶矩放入特征数组
    # 三阶矩 （斜度 skewness）
    h_skewness = np.mean(abs(h - h.mean()) ** 3)
    s_skewness = np.mean(abs(s - s.mean()) ** 3)
    v_skewness = np.mean(abs(v - v.mean()) ** 3)
    h_thirdMoment = h_skewness ** (1. / 3)
    s_thirdMoment = s_skewness ** (1. / 3)
    v_thirdMoment = v_skewness ** (1. / 3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])  # 三阶矩放入特征数组
    return color_feature

# 定义灰度共生矩阵参数
distances = [1]
angles = [np.pi/2]

def Maxgray(filename):
    img = cv2.imread(filename)  # 读一张彩色图片
    if img is None:
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BGR空间转换为灰度空间
    glcm = graycomatrix(img_gray, distances, angles, levels=256, symmetric=True, normed=True)
    # 计算统计特征
    glcm_features = []
    # 定义要计算的统计特征"对比度、不相似度、能量、相关性"
    properties = ['contrast', 'dissimilarity', 'energy', 'correlation']
    for prop in properties:
        prop_value = graycoprops(glcm, prop).flatten()
        glcm_features.extend(prop_value)
    return glcm_features

def multi(filename):
    color_features = color_moments(filename)
    glcm_feature = Maxgray(filename)
    return np.concatenate(color_features,glcm_feature)

def extract_multi(image_folder,output_file):
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
                glcm_features = Maxgray(image_path)
                colorm_features = color_moments(image_path)
                features = np.concatenate((colorm_features, glcm_features))
                print(image_path)
                writer.writerow([image_path]+features.tolist())
    print("multi特征提取完毕")

def search_multi(search_image,output_file):
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

    #读取并计算搜素图片的特征
    search_glcm_features = Maxgray(search_image)
    search_colorm_features = color_moments(search_image)
    search_feature = np.concatenate((search_colorm_features, search_glcm_features)).reshape(1, -1)

    # # 记录检索开始时间
    # start_time = time.time()
    distances = np.linalg.norm(features-search_feature,axis=1)
    # 获取排序后的索引
    sorted_indices = np.argsort(distances)
    top_n = 10
    return [(image_paths[i], distances[i]) for i in sorted_indices[:top_n]]
    # # 记录检索结束时间
    # end_time = time.time()
    # # 计算并打印检索时间
    # search_time = end_time - start_time
    # print(f"检索时间: {search_time:.4f} 秒")
    # # 检索结果
    # top_n = 10  # 选择前5个最相似的图像
    # for i in range(min(top_n, len(sorted_indices))):
    #     index = sorted_indices[i]
    #     print(f"Match {i + 1}: {image_paths[index]} with distance {distances[index]}")

def test_path(image_folder,image_folder2,output_file):
    #获取图像文件列表
    image_files = os.listdir(image_folder)
    image_files2 = os.listdir(image_folder2)
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        #遍历图像文件列表
        for image_file in image_files:
            if image_file.endswith('.jpg'):
                #读取图像
                image_path = os.path.join(image_folder,image_file)
                writer.writerow([image_path])
        for image_file in image_files2:
            if image_file.endswith('.jpg'):
                #读取图像
                image_path2 = os.path.join(image_folder2,image_file)
                writer.writerow([image_path2])
    print("图片提取完毕")