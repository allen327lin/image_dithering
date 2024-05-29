"""
MIT License

Copyright (c) 2024 allen327lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def show_img(title, img, save_img=False):    # 顯示圖片
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)    # 創建視窗
    cv2.resizeWindow(title, 400, 400)    # 設定視窗大小
    cv2.imshow(title, img)    # 顯示圖片
    if save_img:
        cv2.imwrite("./photos/results/" + title + ".png", img)    # 儲存圖片
    return 0


def convert_to_0_and_255(image_8_bit):
    # 將>=127的地方設成255，<127的地方設成0
    _, image_1_bit = cv2.threshold(image_8_bit, 127, 255, cv2.THRESH_BINARY)
    return image_1_bit


def compare_ssim(img1, img2):
    score, diff = ssim(img1, img2, full=True)    # 比對img1和img2，返回結構相似性的圖
    diff = (diff * 255).astype("uint8")    # 把值拉到0~255之間，並設定格式是uint8，確保是圖片的格式
    return score, diff


def normalization(arr):
    min_v = np.min(arr)
    max_v = np.max(arr)
    arr = (arr - min_v) / (max_v - min_v)    # 正規化公式
    arr = (arr * 255).astype(np.uint8)
    return arr
