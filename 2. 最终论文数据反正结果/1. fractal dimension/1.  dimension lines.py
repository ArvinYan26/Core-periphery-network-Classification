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
viral_pneumonia = []
covid_19 = []

min_value, max_value, steps = 100, 160, 10
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

def binary_image(image, data):
    """

    :param image:
    :param data:
    :return:
    """
    for i in Threshold:
        #img1 = []
        #img1.append(image1)
        #for i in Threshold:
        #threshold = 120       #ˈθreʃhoʊld
        #print("i:", i)
        ret, b_img = cv2.threshold(image, i, 255, cv2.THRESH_BINARY)
        data.append(b_img)

    return data

def convertjpg(pngfile, data):

    print("pngfile:", pngfile)
    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,
    #因为opencv读取图片方式和matplot不一样，所以显示结果不同，所以不再显示原图，直接显示灰度图和二值图像
    #img = cv2.imread(pngfile) #读取为灰度图,
    #data.append(img) #添加原图
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data.append(img)
    new_data = binary_image(img, data)

    return new_data

def draw_graph(data, count):
    plt.figure()
    for i in range(len(data)):
        #print("i:", i)
        #plt.annotate('covid-19', xy=(0, 0), xytext=(1, 1))
        plt.subplot(count, len(data)/count, i+1)
        plt.imshow(data[i], "gray") #这是因为我们还是直接使用plt显示图像，它默认使用三通道显示图像，需要我们在可视化函数里多指定一个参数

        """
        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置纵坐标字体信息
        plt.ylabel("Number of blocks", fontsize=20)
        """
        #设置x轴刻度显示值
        #参数一：中点坐标
        #参数二：显示值
        """
        plt.xticks([], fontproperties='Times New Roman', size=20)
        plt.xlabel("Block size", fontsize=20)
        """
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
for pngfile in glob.glob("C:/Users/Arvin Yan/Desktop/COVID-19-c/NORMAL1/*.png"):
    normal = convertjpg(pngfile, normal)
    count0 += 1
    if count0 == 4:
        break
print(len(normal))


"""
count1 = 0
for pngfile in glob.glob(r"C:/Users/Arvin Yan/Desktop/COVID-19-c/Viral Pneumonia/*.png"):
    viral_pneumonia = convertjpg(pngfile, viral_pneumonia)
    count1 += 1
    if count1 == 6:
        break
print(len(viral_pneumonia))
"""

count2 = 0
for pngfile in glob.glob("C:/Users/Arvin Yan/Desktop/COVID-19-c/COVID-191/*.png"):
    covid_19 = convertjpg(pngfile, covid_19)
    count2 += 1
    if count2 == 4:
        break
print(len(covid_19))

"""
#画灰度像素直方图
whole_historgram(normal, count0)
#whole_historgram(viral_pneumonia, count1)
whole_historgram(covid_19, count2)
"""

# #显示不同阈值的二值图像
draw_graph(normal, count0)
# #draw_graph(viral_pneumonia, count1)
draw_graph(covid_19, count2)

#获取维度折线图数据
print("len:", len(normal))
D = get_line_chart_data(normal, count0)
#D1 = get_line_chart_data(viral_pneumonia, count1)
D2 = get_line_chart_data(covid_19, count2)

#画维度折线图
plt.figure("Fractal Demension", figsize=(15, 12))
#plt.rcParams['font.sans-serif']=['SimHei'] #用来显示中文
"""
plt.plot(Threshold, D[0], color="#00ff00", label="normal0")
plt.plot(Threshold, D[1], color="#00ff5f", label="normal1")
plt.plot(Threshold, D[2], color="#00ff87", label="normal2")
plt.plot(Threshold, D[3], color="#00ffaf", label="normal3")
"""
"""
plt.plot(Threshold, D[0], "o-", color="#ff0000", label="a", linestyle='-')
plt.plot(Threshold, D[1], "^-", color="#ff005f", label="b", linestyle='--')
plt.plot(Threshold, D[2], "s-", color="#ff0087", label="c", linestyle='-.')
plt.plot(Threshold, D[3], "x-", color="#ff00af", label="d", linestyle=':')
#plt.plot(Threshold,D[4], color="#d70000", label="normal3")
#plt.plot(Threshold,D[5], color="#d70057", label="normal3")
"""
plt.plot(Threshold, D[0], "o-", color="#ff0000", label="(a)", markersize=10)
plt.plot(Threshold, D[1], "^-", color="#ff0000", label="(b)", markersize=10)
plt.plot(Threshold, D[2], "s-", color="#ff0000", label="(c)", markersize=10)
plt.plot(Threshold, D[3], "x-", color="#ff0000", label="(d)", markersize=10)
#plt.plot(Threshold,D[4], color="#d70000", label="normal3")
#plt.plot(Threshold,D[5], color="#d70057", label="normal3")

plt.plot(Threshold, D2[0], "v-", color="#af00ff", label="(e)", markersize=10)
plt.plot(Threshold, D2[1], "*-", color="#af00ff", label="(f)", markersize=10)
plt.plot(Threshold, D2[2], "+-", color="#af00ff", label="(g)", markersize=10)
plt.plot(Threshold, D2[3], "d-", color="#af00ff", label="(h)", markersize=10)
#plt.plot(Threshold, D2[4], color="#8700ff", label="covid3")
#plt.plot(Threshold, D2[5], color="#875fff", label="covid3")

#handlelength:图例线的长度, borderpad：图例窗口大小, labelspacing：label大小， fontsize：图例字体大小
plt.legend(loc="lower right", handlelength=4, borderpad=2, labelspacing=2, fontsize=12)

plt.yticks(size=15) #设置纵坐标字体信息
#plt.ylabel("Desmension", fontsize=20)

#设置x轴刻度显示值
#参数一：中点坐标
#参数二：显示值
plt.xticks(size=15)
#plt.xlabel("Thershold", fontsize=20)

#plt.xlabel("二值图像阈值", size=20)
#plt.ylabel("分形维数", size=20)
plt.xlabel("Thershold", size=20)
plt.ylabel("Fractral Dimension", size=20)
plt.show()



