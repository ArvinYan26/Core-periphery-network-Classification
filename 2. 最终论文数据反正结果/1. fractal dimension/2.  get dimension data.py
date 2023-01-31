import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()
#data = np.empty(shape=[0, 2916], dtype=int)


#存储处理后的图像
normal = []
#viral_pneumonia = []
covid_19 = []

min_value, max_value, steps = 100, 170, 10
Threshold = [x for x in range(min_value, max_value, steps)]

def whole_historgram(data, count):
    his_img = []
    plt.figure()
    for i in range(len(data)):
        if i % (len(data)/count) == 0: #找到原灰度图像，画出直方图
            his_img.append(data[i])
    #print(len(his_img), data[0])
    for i in range(len(his_img)):
        #print(i, his_img[i])
        #if i == 3:
            #print(his_img[i].ravel())
        plt.hist(his_img[i].ravel(), 256, [0, 256])
        plt.subplot(count/2, 2, i + 1)
        plt.xlabel("Pixel gray value")
        plt.ylabel("Number of pixels")
    plt.show()


def convertjpg(pngfile, data):

    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,
    #因为opencv读取图片方式和matplot不一样，所以显示结果不同，所以不再显示原图，直接显示灰度图和二值图像
    #img = cv2.imread(pngfile) #读取为灰度图,
    #data.append(img) #添加原图
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #data.append(img)
    eac_d = []
    for i in Threshold:
        ret, b_img = cv2.threshold(img, i, 255, cv2.THRESH_BINARY) #二值图像，返回 i:是阈值
        d = fractal_dimension(b_img, i)
        eac_d.append(d)
    data.append(eac_d)
    #new_data = binary_image(img, data)

    return data

def draw_graph(data, count):
    plt.figure()
    for i in range(len(data)):
        #print("i:", i)
        #plt.annotate('covid-19', xy=(0, 0), xytext=(1, 1))
        plt.subplot(count, len(data)/count0, i+1)
        plt.imshow(data[i], "gray") #这是因为我们还是直接使用plt显示图像，它默认使用三通道显示图像，需要我们在可视化函数里多指定一个参数
        plt.xticks([])
        plt.yticks([])
    plt.show()




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

def get_line_chart_data(data, count):
    gray_img = []
    for n in range(len(data)):
        if n % (len(data) / count) == 0:
            gray_img.append(data[n])
    print("len_gry:", len(gray_img))
    l = []
    for j in range(len(gray_img)):
        x = []
        for i in Threshold:
            y = fractal_dimension(gray_img[j], i)
            x.append(y)
        l.append(x)

    return l


#读取图片数据，转化为矩阵
count0 = 0
for pngfile in glob.glob("C:/Users/Arvin Yan/Desktop/COVID-19-c/NORMAL/*.png"):
    normal = convertjpg(pngfile, normal)
    count0 += 1
    if count0 == 150:
        break
print(len(normal))


"""
count1 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/Viral Pneumonia/*.png"):
    viral_pneumonia = convertjpg(pngfile, viral_pneumonia)
    count1 += 1
    if count1 == 6:
        break
print(len(viral_pneumonia))
"""

count2 = 0
for pngfile in glob.glob("C:/Users/Arvin Yan/Desktop/COVID-19-c/COVID-19/*.png"):
    covid_19 = convertjpg(pngfile, covid_19)
    count2 += 1
    if count2 == 150:
        break
print(len(covid_19))

"""
#画灰度像素直方图
whole_historgram(normal, count0)
#whole_historgram(viral_pneumonia, count1)
whole_historgram(covid_19, count2)
"""

"""
#显示不同阈值的二值图像
draw_graph(normal, count0)
#draw_graph(viral_pneumonia, count1)
draw_graph(covid_19, count2)
"""


#获取维度折线图数据
#print("len:", len(normal))
#D = get_line_chart_data(normal, count0)
#D1 = get_line_chart_data(viral_pneumonia, count1)
#D1 = get_line_chart_data(covid_19, count2)



data0 = np.array(normal)
data1 = np.array(covid_19)
data = np.vstack((normal, covid_19))
print(len(data))

save = pd.DataFrame(data)
#保存数据与之范围是100-150，阈值间隔是10.
save.to_csv(r"C:/Users/Yan/Desktop/dimension_100_150_10.csv", index=False, header=True)

end_time = time.time()
time = end_time-start_time
print("time:", time)

