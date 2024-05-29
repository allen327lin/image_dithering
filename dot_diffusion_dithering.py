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

def dot_diffusion_dithering(image, diffusion_matrix):
    height, width = image.shape
    grid_size = diffusion_matrix.shape[0]
    dithered_image = np.zeros_like(image, dtype=np.uint8)
    error = np.zeros_like(image, dtype=np.float32)
    diffusion_matrix = diffusion_matrix.astype(np.float32) / np.sum(diffusion_matrix)

    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            for dy in range(grid_size):
                for dx in range(grid_size):
                    yy = y + dy
                    xx = x + dx
                    if yy >= height or xx >= width:
                        continue

                    old_pixel = image[yy, xx] + error[yy, xx]
                    new_pixel = 255 if old_pixel > 128 else 0
                    dithered_image[yy, xx] = new_pixel
                    quant_error = old_pixel - new_pixel

                    for wy in range(grid_size):
                        for wx in range(grid_size):
                            if wy + yy >= height or wx + xx >= width:
                                continue
                            error[wy + yy, wx + xx] += quant_error * diffusion_matrix[wy, wx]

    return dithered_image

# 定义扩散矩阵 (例如 3x3 网格)
diffusion_matrix = np.array([
    [0, 0, 0, 7, 5],
    [3, 5, 7, 5, 3],
    [1, 3, 5, 3, 1],
])

# 读取灰度图像
image = cv2.imread('photos/profile_photo_1025.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('Image file not found.')

# 应用 Dot Diffusion Dithering
start_t = time()
dithered_image = dot_diffusion_dithering(image, diffusion_matrix)
end_t = time()
print("執行時間：" + str(round(end_t - start_t, 3)) + "s")

# 保存处理后的图像
cv2.imwrite('photos/dot_diffusion_dithering.png', dithered_image)

# 显示处理后的图像
show_img('Dithered Image Dot Diffusion', dithered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
