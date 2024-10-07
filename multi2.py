from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import numpy as np
import cv2
import os
import csv

def sift(filename):
    # 读取图像并转换为灰度图像
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用SIFT特征检测算法提取关键点和特征描述符
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

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
                color_features = color_moments(image_path)
                descriptors = sift(image_path)

                if descriptors is not None:
                    histogram, _ = np.histogram(
                        kmeans.predict(descriptors), bins=np.arange(kmeans.n_clusters + 1)
                    )
                    histogram = histogram * 0.02
                    row = [str(image_path)]+ color_features + histogram.tolist()
                    writer.writerow(row)

                print("multi2创造" + image_path)

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

    color_features = color_moments(query_image_path)
    descriptors = sift(query_image_path)

    if descriptors is not None:
        query_histogram, _ = np.histogram(
            kmeans.predict(descriptors), bins=np.arange(kmeans.n_clusters + 1)
        )
        query_histogram = query_histogram.flatten()
        color_features = np.array(color_features)
        color_features = color_features
        query_histogram = query_histogram * 0.02
        query_features = np.hstack((color_features ,query_histogram))

        distances = cdist([query_features], histograms, 'euclidean')[0]
        sorted_indices = np.argsort(distances)

        top_n = 10
        # for i in range(min(top_n, len(sorted_indices))):
        #     index = sorted_indices[i]
        #     print(f"Match {i + 1}: {image_paths[index]} with distance {distances[index]}")
        return [(image_paths[i], distances[i]) for i in sorted_indices[:top_n]]