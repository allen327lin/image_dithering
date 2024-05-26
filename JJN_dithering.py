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

def jarvis_judice_ninke_dither(image):
    # 将图像转换为浮点型，以便进行误差扩散
    pixels = image.astype(float)
    height, width = pixels.shape

    # 遍历每一个像素
    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            pixels[y, x] = new_pixel
            error = old_pixel - new_pixel

            # 分散误差
            if x + 1 < width:
                pixels[y, x + 1] += error * 7 / 48
            if x + 2 < width:
                pixels[y, x + 2] += error * 5 / 48

            if y + 1 < height:
                if x - 2 >= 0:
                    pixels[y + 1, x - 2] += error * 3 / 48
                if x - 1 >= 0:
                    pixels[y + 1, x - 1] += error * 5 / 48
                pixels[y + 1, x] += error * 7 / 48
                if x + 1 < width:
                    pixels[y + 1, x + 1] += error * 5 / 48
                if x + 2 < width:
                    pixels[y + 1, x + 2] += error * 3 / 48

            if y + 2 < height:
                if x - 2 >= 0:
                    pixels[y + 2, x - 2] += error * 1 / 48
                if x - 1 >= 0:
                    pixels[y + 2, x - 1] += error * 3 / 48
                pixels[y + 2, x] += error * 5 / 48
                if x + 1 < width:
                    pixels[y + 2, x + 1] += error * 3 / 48
                if x + 2 < width:
                    pixels[y + 2, x + 2] += error * 1 / 48

    # 将像素值限制在0到255之间，并转换回8位无符号整数类型
    return np.clip(pixels, 0, 255).astype(np.uint8)

# 读取灰度图像
image = cv2.imread('photos/profile_photo_1025.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('Image file not found.')

# 应用 Jarvis, Judice, and Ninke Dithering
dithered_image = jarvis_judice_ninke_dither(image)

# 保存处理后的图像
cv2.imwrite('photos/JJN_dithering.png', dithered_image)

# 显示原图和处理后的图像
show_img('Original Image', image)
show_img('Dithered Image', dithered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
