import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import os
import csv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import time


# def resize_image(image,max_width,max_height):
#     # 获取原始图片的宽度和高度
#     height, width = image.shape[:2]
#     # 计算缩放比例
#     scale = min(max_width / width, max_height / height)
#     # 根据缩放比例计算新的宽度和高度
#     new_width = int(width * scale)
#     new_height = int(height * scale)
#     # 缩小图片
#     resized_image = cv2.resize(image, (new_width, new_height))
#     return resized_image
#
# # 设置最大宽度和最大高度
# max_width = 800
# max_height = 800


def sift(filename):
    # 读取图像并转换为灰度图像
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用SIFT特征检测算法提取关键点和特征描述符
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def build_visual_vocabulary(image_folder, n_clusters):
    descriptors_list = []
    image_files = os.listdir(image_folder)

    for image_file in image_files:
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_file)
            descriptors = sift(image_path)
            if descriptors is not None:
                descriptors_list.append(descriptors)
            print("bow"+image_path)

    all_descriptors = np.vstack(descriptors_list)
    print("打包成功")
    kmeans =KMeans(n_clusters=n_clusters)
    kmeans.fit(all_descriptors)
    print("Kmeans成功")
    return kmeans


def create_histograms(image_folder, kmeans, output_file):
    image_files = os.listdir(image_folder)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for image_file in image_files:
            if image_file.endswith('.jpg'):
                image_path = os.path.join(image_folder, image_file)
                descriptors = sift(image_path)
                if descriptors is not None:
                    histogram, _ = np.histogram(
                        kmeans.predict(descriptors), bins=np.arange(kmeans.n_clusters + 1)
                    )
                    row = [image_path] + histogram.tolist()
                    writer.writerow(row)
                print("创造"+image_path)



# def extract_sift_pca(image_folder, output_file, n_components=40):
#     all_descriptors = []
#     image_paths = []
#     # 获取图像文件列表
#     image_files = os.listdir(image_folder)
#     # 遍历图像文件列表
#     for image_file in image_files:
#         if image_file.endswith('.jpg'):
#             # 读取图像
#             image_path = os.path.join(image_folder, image_file)
#             # 提取特征
#             descriptors = sift(image_path)
#             print("提取"+image_path)
#             all_descriptors.append(descriptors)
#             image_paths.append(image_path)
#
#     # 将所有特征堆叠起来进行PCA
#     all_descriptors = np.vstack(all_descriptors)
#     print("折叠PCA")
#
#     pca = PCA(n_components=n_components)
#     reduced_descriptors = pca.fit_transform(all_descriptors)
#     print("PCA成功")
#
#     with open(output_file, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         start_idx = 0
#         for image_path in image_paths:
#             num_descriptors = len(sift(image_path))
#             image_descriptors = reduced_descriptors[start_idx:start_idx + num_descriptors]
#             for feature in image_descriptors:
#                 row = [image_path] + feature.tolist()
#                 writer.writerow(row)
#             start_idx += num_descriptors
#             print(image_path)
#
#         print("特征提取和降维完毕")
def save_kmeans_model(kmeans, filename):
    with open(filename, 'wb') as f:
        pickle.dump(kmeans, f)

    print("保存模型成功")

def load_kmeans_model(filename):
    with open(filename, 'rb') as f:
        kmeans = pickle.load(f)
    return kmeans


def search_image(query_image_path, kmeans, output_file):
    histograms = []
    image_paths = []

    with open(output_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            image_paths.append(row[0])
            histograms.append([float(val) for val in row[1:]])

    histograms=np.array(histograms)

    descriptors = sift(query_image_path)

    if descriptors is not None:
        query_histogram, _ = np.histogram(
            kmeans.predict(descriptors), bins=np.arange(kmeans.n_clusters + 1)
        )
        distances = cdist([query_histogram], histograms, 'euclidean')[0]
        sorted_indices = np.argsort(distances)

        top_n = 10
        # for i in range(min(top_n, len(sorted_indices))):
        #     index = sorted_indices[i]
        #     print(f"Match {i + 1}: {image_paths[index]} with distance {distances[index]}")
        return [(image_paths[i], distances[i]) for i in sorted_indices[:top_n]]
# def search_sift(search_image, output_file):
#     features = []
#     image_paths = []
#     with open(output_file, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             image_path = row[0]
#             # 将字符串形式的列表转换为浮点数数组
#             feature = np.array([float(val) for val in row[1:]])
#             image_paths.append(image_path)
#             features.append(feature)
#
#     features = np.array(features)

    # start_time = time.time()

    # search_descriptors = sift(search_image)
    # search_descriptors = pca.transform(search_descriptors)
    # # 计算查询特征的平均值
    # search_feature_mean = np.mean(search_descriptors, axis=0)
    # # 计算欧氏距离并排序
    # distances = np.linalg.norm(features - search_feature_mean, axis=1)
    # sorted_indices = np.argsort(distances)
    # top_n = 10
    # return [(image_paths[i], distances[i]) for i in sorted_indices[:top_n]]

    #验证
    # end_time = time.time()
    # search_time = end_time - start_time
    # print(f"检索时间: {search_time:.4f} 秒")
    #
    # top_n = 10
    # for i in range(min(top_n, len(sorted_indices))):
    #     index = sorted_indices[i]
    #     print(f"Match {i + 1}: {image_paths[index]} with distance {distances[index]}")
