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

import cv2
import numpy as np
from utils import show_img
from time import time
from convert_to_0_and_255 import convert_8_to_1_bit

def floyd_steinberg_dither(image):
    pixels = image.astype(float)
    height, width = pixels.shape

    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            pixels[y, x] = new_pixel
            error = old_pixel - new_pixel

            # 分散誤差到周圍像素
            if x + 1 < width:
                pixels[y, x + 1] += error * 7 / 16
            if y + 1 < height:
                if x - 1 >= 0:
                    pixels[y + 1, x - 1] += error * 3 / 16
                pixels[y + 1, x] += error * 5 / 16
                if x + 1 < width:
                    pixels[y + 1, x + 1] += error * 1 / 16

    # 將像素值壓在0到255之間，並轉成8-bit整數類型使符合照片格式
    return np.clip(pixels, 0, 255).astype(np.uint8)

# 讀入原圖
image = cv2.imread('photos/profile_photo_1025.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('Image file not found.')

# 執行 Floyd-Steinberg Dithering
start_t = time()
dithered_image = floyd_steinberg_dither(image)
end_t = time()
print("執行時間：" + str(round(end_t - start_t, 3)) + "s")

# 儲存結果圖
cv2.imwrite('photos/floyd_steinberg_dithering.png', dithered_image)

# 顯示原圖與結果圖
show_img("Original Image", image)
show_img("Dithered Image", dithered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
