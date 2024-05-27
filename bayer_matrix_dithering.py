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

def ordered_dither(image, bayer_matrix):
    bayer_matrix = np.array(bayer_matrix, dtype=float)
    threshold_matrix = (bayer_matrix + 0.5) / (bayer_matrix.size + 1) * 255
    height, width = image.shape
    dithered_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            i = y % bayer_matrix.shape[0]
            j = x % bayer_matrix.shape[1]
            if image[y, x] > threshold_matrix[i, j]:
                dithered_image[y, x] = 255
            else:
                dithered_image[y, x] = 0

    return dithered_image

def generate_bayer_matrix(n):
    if n == 1:
        return np.array([[0]])
    else:
        smaller_matrix = generate_bayer_matrix(n // 2)
        size = smaller_matrix.shape[0]
        new_matrix = np.zeros((n, n), dtype=int)
        new_matrix[0:size, 0:size] = 4 * smaller_matrix
        new_matrix[0:size, size:n] = 4 * smaller_matrix + 2
        new_matrix[size:n, 0:size] = 4 * smaller_matrix + 3
        new_matrix[size:n, size:n] = 4 * smaller_matrix + 1
        return new_matrix

bayer_2x2 = generate_bayer_matrix(2)
bayer_4x4 = generate_bayer_matrix(4)
bayer_8x8 = generate_bayer_matrix(8)
bayer_16x16 = generate_bayer_matrix(16)

# bayer_2x2 = [
#     [0, 2],
#     [3, 1]
# ]

# bayer_4x4 = [
#     [ 0,  8,  2, 10],
#     [12,  4, 14,  6],
#     [ 3, 11,  1,  9],
#     [15,  7, 13,  5]
# ]

# bayer_8x8 = [
#     [ 0, 32,  8, 40,  2, 34, 10, 42],
#     [48, 16, 56, 24, 50, 18, 58, 26],
#     [12, 44,  4, 36, 14, 46,  6, 38],
#     [60, 28, 52, 20, 62, 30, 54, 22],
#     [ 3, 35, 11, 43,  1, 33,  9, 41],
#     [51, 19, 59, 27, 49, 17, 57, 25],
#     [15, 47,  7, 39, 13, 45,  5, 37],
#     [63, 31, 55, 23, 61, 29, 53, 21]
# ]

# 讀入原圖
image = cv2.imread('photos/profile_photo_1025.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError('Image file not found.')

# 執行 Bayer Matrix (Ordered Dithering)
dithered_image_2x2 = ordered_dither(image, bayer_2x2)
dithered_image_4x4 = ordered_dither(image, bayer_4x4)
dithered_image_8x8 = ordered_dither(image, bayer_8x8)

# 儲存結果圖
cv2.imwrite('photos/bayer_matrix_dithering_2x2.png', dithered_image_2x2)
cv2.imwrite('photos/bayer_matrix_dithering_4x4.png', dithered_image_4x4)
cv2.imwrite('photos/bayer_matrix_dithering_8x8.png', dithered_image_8x8)

# 顯示原圖與結果圖
show_img('Original Image', image)
show_img('Dithered Image 2x2', dithered_image_2x2)
show_img('Dithered Image 4x4', dithered_image_4x4)
show_img('Dithered Image 8x8', dithered_image_8x8)
cv2.waitKey(0)
cv2.destroyAllWindows()
