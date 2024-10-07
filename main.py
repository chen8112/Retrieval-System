import color_M
import color_H
import multi
import GLCM
import SIFT
import multi2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import RetrievalApp


path = "F:\\final\\feature\\gallery"
# path2="F:\\final\\feature\\query"
outfile = "F:\\final\\feature\\sift.csv"
index="F:\\final\\feature\\kmeans.pkl"
# search = "F:\\final\\feature\\query\\25_c2s2_000826_7.jpg"
search = "F:\\final\\feature\\query\\1_c1s1_000826_1.jpg"
# search = "F:\\final\\feature\\query\\350_c1s1_000826_16.jpg"
# search = "F:\\final\\feature\\query\\305_c1s1_000826_5.jpg"

if __name__ == '__main__':
    # GLCM.extract_glcm(path,outfile)
    # kmeans=SIFT.build_visual_vocabulary(path,n_clusters=100)
    # SIFT.create_histograms(path,kmeans,outfile)
    # SIFT.save_kmeans_model(kmeans,index)
    # kmeans=SIFT.load_kmeans_model(index)
    # results = SIFT.search_image(search, kmeans,"sift.csv")
    # print(results)
    app = RetrievalApp.ImageRetrievalApp()
    app.mainloop()