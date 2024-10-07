import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.pairwise import euclidean_distances
import os
import csv
import time

# 定义灰度共生矩阵参数
distances = [1,2,3]
angles = [0,np.pi/4,np.pi/2]


def Maxgray(filename):
    img = cv2.imread(filename)  # 读一张彩色图片
    if img is None:
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BGR空间转换为灰度空间
    glcm = graycomatrix(img_gray, distances, angles, levels=256, symmetric=True, normed=True)
    # 计算统计特征
    features = []
    # 明确定义要计算的统计特征"对比度、不相似度、能量、相关性"的顺序
    properties = ['contrast', 'dissimilarity', 'energy', 'correlation']
    for prop in properties:
        prop_value = graycoprops(glcm, prop).flatten()
        features.extend(prop_value)
    return features


def extract_glcm(image_folder,output_file):
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
                features = Maxgray(image_path)
                # print(image_path, features)
                writer.writerow([image_path]+features)
    print("glcm特征提取完毕")

def search_glcm(search_image,output_file):
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
    search_feature = Maxgray(search_image)
    search_feature = np.array(search_feature)
    print(search_feature)

    # start_time = time.time()
    distances = np.linalg.norm(features - search_feature, axis=1)
    sorted_indices = np.argsort(distances)
    top_n = 10
    return [(image_paths[i], distances[i]) for i in sorted_indices[:top_n]]

    # end_time = time.time()
    #
    # search_time = end_time - start_time
    # print(f"检索时间: {search_time:.4f} 秒")
    #
    # top_n = 10
    # for i in range(min(top_n, len(sorted_indices))):
    #     index = sorted_indices[i]
    #     print(f"Match {i + 1}: {image_paths[index]} with distance {distances[index]}")

    # # 计算相似度并排序
    # similarities = []
    # for feature in features:
    #     similarity = match_features(search_feature, feature)
    #     similarities.append(similarity)
    #
    # sorted_indices = np.argsort(similarities)[::-1]
    # sorted_similarities = sorted(similarities, reverse=True)
    # # 获取前10个最相似的图像及其相似度
    #
    # top_matches = []
    # for i in range(10):
    #     index = sorted_indices[i]
    #     similarity = sorted_similarities[i]
    #     top_matches.append((index, similarity))
    #
    # return top_matches

