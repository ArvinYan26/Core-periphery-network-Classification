import os.path
import glob
import cv2
import numpy as np
import pandas as pd
import time

start_time = time.time()
min_value, max_value, steps = 100, 170, 10
Threshold = [x for x in range(min_value, max_value, steps)]
def convertjpg(pngfile, class_num, data, data_target, width=512, height=512):
    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,
    dst = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)   #interpola：双线性插值方式
    dst = fft(dst) #进行傅里叶变化
    dst = dst.reshape(1, -1)
    data.append(dst[0])
    data_target.append(class_num)

# def convertjpg(pngfile, class_num, data, data_target, width=512, height=512):
#     img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,
#     # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)   #interpola：双线性插值方式
#     # dst = fft(dst) #进行傅里叶变化
#     # dst = dst.reshape(1, -1)
#
#
#     eac_d = []
#     for i in Threshold:
#         ret, b_img = cv2.threshold(img, i, 255, cv2.THRESH_BINARY)  # 二值图像，返回 i:是阈值
#         d = fractal_dimension(b_img, i)
#         eac_d.append(d)
#     # print("eac_d:", eac_d)
#     data.append(eac_d)
#
#
#     # print("dst:", dst)
#
#     # data.append(dst[0])
#
#
#     data_target.append(class_num)

def fractal_dimension(Z, threshold):
    # Only for 2d image
    assert(len(Z.shape) == 2) #assert ： 判断语句，如果是二维图像就执行，不是的话，直接报错，不再执行

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        #axis=0, 按列计算， 1：按行计算
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)
    #plt.draw()
    #plt.show()

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    return -coeffs[0]


def fft(img):
    #快速傅里叶变换算法得到频率分布
    f = np.fft.fft2(img)

    #默认结果中心点位置是在左上角,
    #调用fftshift()函数转移到中间位置
    fshift = np.fft.fftshift(f)

    #fft结果是复数, 其绝对值结果是振幅
    fimg = np.log(np.abs(fshift))

    return fimg


def get_data():
    data0 = []
    data1 = []
    data2 = []
    data_target0 = []
    data_target1 = []
    data_target2 = []

    #读取图片数据，转化为矩阵
    count0 = 0
    for pngfile in glob.glob(r"C:/Users/Arvin Yan/Desktop/COVID-19-c/NORMAL/*.png"):
        convertjpg(pngfile, 0, data0, data_target0)
        count0 += 1
        if count0 == 150:
            break


    # count = 0
    # for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/Viral Pneumonia/*.png"):
    #     convertjpg(pngfile, 1, data1, data_target1)
    #     count += 1
    #     if count == 150:
    #         break
    count1 = 0

    for pngfile in glob.glob(r"C:/Users/Arvin Yan/Desktop/COVID-19-c/COVID-19/*.png"):
        convertjpg(pngfile, 1, data2, data_target2)
        count1 += 1
        if count1 == 150:
            break

    data0 = np.array(data0)
    #data1 = np.array(data1)
    data2 = np.array(data2)

    data = np.vstack((data0, data2))
    # print("data:", data)
    #data = np.vstack((data0, data2))

    data_target = np.array(data_target0 + data_target1 + data_target2)

    """
    save = pd.DataFrame(data0)
    save.to_csv(r"C:/Users/Yan/Desktop/data.csv", index=False, header=True)
    
    data_target = pd.DataFrame(data_target)
    data_target.to_csv(r"C:/Users/Yan/Desktop/data_taget.csv", index=False, header=True)
    """
    return data, data_target


#end_time = time.time()
#time = end_time - start_time
#print("time:", time)



