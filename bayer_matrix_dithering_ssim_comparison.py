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
from utils import compare_ssim, show_img

# 讀入gray scale圖片
img1 = cv2.imread("./photos/profile_photo_1025.jpg", 0)
img2 = cv2.imread("./photos/bayer_matrix_dithering_2x2.png", 0)
img3 = cv2.imread("./photos/bayer_matrix_dithering_4x4.png", 0)
img4 = cv2.imread("./photos/bayer_matrix_dithering_8x8.png", 0)
img5 = cv2.imread("./photos/JJN_dithering.png", 0)
img6 = cv2.imread("./photos/dot_diffusion_dithering.png", 0)

# 用SSIM指標比對原圖與Dithering結果圖
score, diff = compare_ssim(img1, img1)
score2, diff2 = compare_ssim(img1, img2)
score3, diff3 = compare_ssim(img1, img3)
score4, diff4 = compare_ssim(img1, img4)
score5, diff5 = compare_ssim(img1, img5)
score6, diff6 = compare_ssim(img1, img6)
print(score, score2, score3, score4, score5, score6, sep="\n")

# 顯示原圖與Dithering的相似處
show_img("difference", diff)
show_img("difference2", diff2)
show_img("difference3", diff3)
show_img("difference4", diff4)
show_img("difference5", diff5)
show_img("difference6", diff6)

# 儲存結構相似圖
cv2.imwrite("./photos/bayer_matrix_dithering_2x2_ssim.png", diff2)
cv2.imwrite("./photos/bayer_matrix_dithering_4x4_ssim.png", diff3)
cv2.imwrite("./photos/bayer_matrix_dithering_8x8_ssim.png", diff4)
cv2.imwrite("./photos/JJN_dithering_ssim.png", diff5)
cv2.imwrite("./photos/dot_diffusion_dithering_ssim.png", diff6)

cv2.waitKey(0)
cv2.destroyAllWindows()
