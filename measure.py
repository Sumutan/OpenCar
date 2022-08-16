"""
该文件提供在图像中根据openopse给出的点位以及deeplab给出的边界绘制各种点位以及延长线的函数
各种操作基于opencv-python库
主要自定义了一个Img图像类（面向对象设计），然后在此基础上用内部的函数对内部的图像数据做处理
"""
import cv2 as cv
import numpy as np
import array
import matplotlib.pyplot as plt
import os
import math
import time


class Img:
    """
        自定义图片类,用于封装我们做的操作
        因为涉及到的函数动作可能比较多,所以这次打算用OOP编程
    """

    def __init__(self, img_path, type='gray'):
        if type == 'gray':
            self.img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        elif type == 'color':
            self.img = cv.imread(img_path, cv.IMREAD_COLOR)
        else:
            print("init type error")
        self.path = img_path

    def img_show(self):  # 显示图片与相关参数
        print("path:" + self.path)
        print("shape:" + str(self.img.shape))
        cv.imshow(winname=self.path, mat=self.img)  # 显示图片，第一个参数为图片的标题
        cv.waitKey()  # 等待图片的关闭，不写这句图片会一闪而过

    def test(self):  # 用于开发过程中方便快速测试的函数,没啥用
        # print(self.img[210,210])
        # print(self.img[210, 290])
        r = 3
        point = [(210, 210), (290, 210)]
        cv.line(self.img, point[0], point[1], (100, 0, 0), 1)

        self.img[210 - r:210 + r, 210 - r:210 + r] = 100
        cv.imshow(winname=self.path, mat=self.img)  # 显示图片，第一个参数为图片的标题
        cv.waitKey()  # 等待图片的关闭，不写这句图片会一闪而过

    def draw_line(self, p1, p2, r=1, color=(100, 0, 0)):  # 画一条线 p1,p2 为(x,y)的元组或列表;r为宽度
        cv.line(self.img, p1, p2, color, r)
        # self.img[210 - r:210 + r, 210 - r:210 + r] = 100
        # cv.imshow(winname=self.path, mat=self.img)  # 显示图片，第一个参数为图片的标题
        # cv.waitKey(0)  # 等待图片的关闭，不写这句图片会一闪而过

    def draw_point(self, point, r=3):  # 画一个点 point为坐标元组,r为点半径
        x, y = point[1], point[0]
        self.img[x - r:x + r, y - r:y + r] = 100

    def draw_Crosspoint(self, point, r=3):  # 画一个点 point为坐标元组,r为点半径
        x, y = point[1], point[0]
        self.img[x, y - r:y + r] = 100
        self.img[x - r:x + r, y] = 100

    def measure(self, p1, p2):  # 通过输入两点位置,得到两点延长先与人体边缘的交点,输出边缘点以及长度
        x1, x2 = p1[0], p2[0]
        y1, y2 = p1[1], p2[1]
        tp1, tp2 = (0, 0), (1, 1)  # tp:Temporary point

        if x1 == x2:
            if y1 == y2:
                print("请勿输入相同的点")
                return

            dy = y2 - y1
            if dy > 0:
                dy = 1
            elif dy < 0:
                dy = -1
            print("dy:" + str(dy))

            x = x1
            y = y2
            while 0 < y < self.img.shape[0]:
                y -= dy
                if self.img[y, x] < 90:
                    tp1 = (x, y)
                    break
            print("tp1:" + str(tp1))

            y = y1
            while 0 < y < self.img.shape[0]:
                y += dy
                if self.img[y, x] < 90:
                    tp2 = (x, y)
                    break
            print("tp2:" + str(tp2))
            length = math.fabs(tp1[1] - tp2[1])
            return tp1, tp2, length
        # assert y1 != y2

        k = (y2 - y1) / (x2 - x1)
        print("k:" + str(k))

        if math.fabs(y2 - y1) > math.fabs(x2 - x1):
            print("纵轴")
            for y in range(y1, 0, -1):
                x = int(x1 - (y1 - y) / k)
                if self.img[y, x] < 90:
                    tp1 = (x, y)
                    break
            print("tp1:" + str(tp1))

            for y in range(y2, self.img.shape[0]):
                x = int(x2 + (y - y2) / k)
                if self.img[y, x] < 90:
                    tp2 = (x, y)
                    break
            print("tp2:" + str(tp2))

            length = math.sqrt((tp1[0] - tp2[0]) ** 2 + (tp1[1] - tp2[1]) ** 2)
            print("length:" + str(length))
            return tp1, tp2, length

        else:
            print("横轴")
            for x in range(x1, 0, -1):
                y = int(y1 - (x1 - x) * k)
                if self.img[y, x] < 90:
                    tp1 = (x, y)
                    break
            print("tp1:" + str(tp1))

            for x in range(x2, self.img.shape[1]):
                y = int(y2 + (x - x2) * k)
                if self.img[y, x] < 90:
                    tp2 = (x, y)
                    break
            print("tp2:" + str(tp2))

            length = math.sqrt((tp1[0] - tp2[0]) ** 2 + (tp1[1] - tp2[1]) ** 2)
            print("length:" + str(length))
            return tp1, tp2, length

    def saveimg(self, savepath):
        cv.imwrite(savepath, self.img)

    def measure_tall(self, p1, p2):  # p1为胸口点，p2为鼻子点,返回身高
        tp1, tp2, length = self.measure(p1, p2)
        if(tp2[1]<tp1[1]):
            tp=tp2
        else:
            tp=tp1
        # print(p2[1])
        return tp  # 返回身高的坐标

    def measure_tall_one_point(self,p1):
        x,y=p1[0],p1[1]
        while self.img[y, x] > 90:
            y-=1
        tp=(x,y)
        return tp

if __name__ == '__main__':
    img_path = "save_img1.jpg"  # 图片路径
    img = Img(img_path)
    # img.img_show()
    p1, p2 = (255, 220), (250, 210)  # 输入一对测试点
    img.draw_point(p1)
    img.draw_point(p2)
    # timg.draw_point(p1)ime_start = time.time()
    tp1, tp2, length = img.measure(p1, p2)  # 测量长度,输出边缘点
    time_end = time.time()
    # print('time cost:', (time_end - time_start) * 1000, 'ms')

    img.draw_line(tp1, tp2)  # 在原图上画线

    img.draw_point(p1)
    img.draw_point(p2)

    img.draw_point(tp1)  # 在原图上画边缘点
    img.draw_point(tp2)
    p2 = img.measure_tall_one_point(p1)

    img.draw_Crosspoint(p2)
    img.img_show()

    # p=(3, 210)
    # print(img.img[:,210])
    # print(len(img.img[:, 210]))
    # img.draw_point(p)
    # img.img_show()

