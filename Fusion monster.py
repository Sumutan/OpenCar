import math
import os

import copy

from src import util
from src.body import Body
from output_processing import candidateright
import time

import cv2
from PIL import Image

from Parents_demo.deeplab import DeeplabV3
from measure import Img
from writedownExcel import writedownlength2Excel

candidate_picture_file = r"images/img_processed/"

input_path = "images/img_processed/"
save_path = "save_img"
excel_path = r"C:\Users\10355\Desktop"

global r_image

if __name__ == "__main__":

    time_start = time.time()

    body_estimation = Body('model/body_pose_model.pth')  # 生成openpose模型
    deeplab = DeeplabV3()
    # hand_estimation = Hand('model/hand_pose_model.pth')
    photopath = os.listdir(candidate_picture_file)      #绑定输入照片文件夹路径
    num_picture = len(photopath)


    ###################################################################由此进入循环

    excelpath = './'
    npicture = 1
    for i in range(num_picture):
        test_image = photopath[i]
        print(test_image)


        oriImg = cv2.imread(candidate_picture_file + test_image)  # B,G,R order
        candidate, subset = body_estimation(oriImg)  # 图片推理，
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # print(candidate)
        # print(subset)
        # print(np.shape(candidate))
        candidateright(candidate, subset)

        ############################################################################
        ############################################################################
        # -------------------------------------------------------------------------#
        #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
        # -------------------------------------------------------------------------#
        # deeplab = DeeplabV3()  #生成deeplab模型   与body_estimation一样不要重复使用########不在循环里
        # ----------------------------------------------------------------------------------------------------------#
        #   mode用于指定测试的模式：
        #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
        #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
        #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
        #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
        #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
        # ----------------------------------------------------------------------------------------------------------#
        mode = "predict"
        # -------------------------------------------------------------------------#
        #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
        #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
        #
        #   count、name_classes仅在mode='predict'时有效
        # -------------------------------------------------------------------------#
        count = False
        name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                        "cow",
                        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                        "tvmonitor"]
        # name_classes    = ["background","cat","dog"]
        # ----------------------------------------------------------------------------------------------------------#
        #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
        #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
        #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
        #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
        #   video_fps           用于保存的视频的fps
        #
        #   video_path、video_save_path和video_fps仅在mode='video'时有效
        #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
        # ----------------------------------------------------------------------------------------------------------#

        # ----------------------------------------------------------------------------------------------------------#
        #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
        #   fps_image_path      用于指定测试的fps图片
        #
        #   test_interval和fps_image_path仅在mode='fps'有效
        # ----------------------------------------------------------------------------------------------------------#
        test_interval = 100
        fps_image_path = "img/street.jpg"
        # -------------------------------------------------------------------------#
        #   dir_origin_path     指定了用于检测的图片的文件夹路径
        #   dir_save_path       指定了检测完图片的保存路径
        #
        #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
        # -------------------------------------------------------------------------#
        dir_origin_path = "img/"
        dir_save_path = "img_out/"
        # -------------------------------------------------------------------------#
        #   simplify            使用Simplify onnx
        #   onnx_save_path      指定了onnx的保存路径
        # -------------------------------------------------------------------------#
        simplify = True
        onnx_save_path = "model_data/models.onnx"

        if mode == "predict":
            '''
            predict.py有几个注意点
            1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
            具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
            2、如果想要保存，利用r_image.save("img.jpg")即可保存。
            3、如果想要原图和分割图不混合，可以把blend参数设置成False。
            4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
            seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
            for c in range(self.num_classes):
                seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
                seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
                seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
            '''

            img = input_path + test_image  # ---》
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
            else:
                r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)  ################图片推理
                r_image.save("img.jpg")
                # r_image.show()

        else:
            raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")

        print(r_image)
        img = Img("img.jpg")
        # img.img_show()

        p0, p1 = (int(candidate[0][0]), int(candidate[0][1])), (int(candidate[10][0]), int(candidate[10][1]))  ###鼻子
        p2, p5 = (int(candidate[2][0]), int(candidate[2][1])), (int(candidate[5][0]), int(candidate[5][1]))  # 左右肩
        p3, p6 = (int(candidate[3][0]), int(candidate[3][1])), (int(candidate[6][0]), int(candidate[6][1]))  # 左右肘
        p4, p7 = (int(candidate[4][0]), int(candidate[4][1])), (int(candidate[7][0]), int(candidate[7][1]))  # 左右手腕
        p8, p11 = (int(candidate[8][0]), int(candidate[8][1])), (int(candidate[11][0]), int(candidate[11][1]))  ###腰围
        p9, p12 = (int(candidate[9][0]), int(candidate[9][1])), (int(candidate[12][0]), int(candidate[12][1]))  ### 左右膝盖
        p10, p13 = (int(candidate[10][0]), int(candidate[10][1])), (
            int(candidate[13][0]), int(candidate[13][1]))  ###左右脚跟
        # p1013=(int(int(candidate[10][0])+int(candidate[13][0]))/2,int(int(candidate[10][1])+int(candidate[13][1]))/2)
        p1013 = (int((candidate[10][0] + candidate[13][0]) / 2), int((candidate[10][1] + candidate[13][1]) / 2))
        print(p2, p5)
        print(p8, p11)
        print(p10, p13)
        print(p1013)
        # time_start = time.time()
        tp0, tp1, length1 = img.measure(p0, p1013)  ##身高
        tp2, tp5, length2 = img.measure(p2, p5)  # 肩宽 测量长度,输出边缘点
        tp8, tp11, length3 = img.measure(p8, p11)  ##臀围
        length4 = int(
            (math.sqrt(pow(candidate[2][0] - candidate[3][0], 2) + pow(candidate[2][1] - candidate[3][1], 2)) +
             math.sqrt(pow(candidate[5][0] - candidate[6][0], 2) + pow(candidate[5][1] - candidate[6][1], 2))) / 2)
        length5 = int(
            (math.sqrt(pow(candidate[3][0] - candidate[4][0], 2) + pow(candidate[3][1] - candidate[4][1], 2)) +
             math.sqrt(pow(candidate[6][0] - candidate[7][0], 2) + pow(candidate[6][1] - candidate[7][1], 2))) / 2)
        length6 = int(
            (math.sqrt(pow(candidate[8][0] - candidate[9][0], 2) + pow(candidate[8][1] - candidate[9][1], 2)) +
             math.sqrt(pow(candidate[11][0] - candidate[12][0], 2) + pow(candidate[11][1] - candidate[12][1], 2))) / 2)
        length7 = int(
            (math.sqrt(pow(candidate[9][0] - candidate[10][0], 2) + pow(candidate[9][1] - candidate[10][1], 2)) +
             math.sqrt(pow(candidate[12][0] - candidate[13][0], 2) + pow(candidate[12][1] - candidate[13][1], 2))) / 2)
        print("身高", length1)
        print("肩宽", length2)
        print("臀宽", length3)
        print("大臂", length4)
        print("小臂", length5)
        print("大腿", length6)
        print("小腿", length7)
        time_end = time.time()
        print('time cost:', (time_end - time_start) * 1000, 'ms')

        # 写入excel
        parameters = [test_image, length1, length2, length3, length4, length5, length6, length7]
        writedownlength2Excel(excelpath, parameters, npicture)  # npicture从1开始
        npicture += 1

        img.draw_line(tp2, tp5)  # 在原图上画线
        img.draw_line(tp8, tp11)
        img.draw_line(tp1, tp0)

        img.draw_point(p2)
        img.draw_point(p5)
        img.draw_point(p8)  # 在原图上画边缘点
        img.draw_point(p11)
        img.draw_point(p1013)

        img.draw_point(tp2)  # 在原图上画边缘点
        img.draw_point(tp5)
        img.draw_point(tp8)  # 在原图上画边缘点
        img.draw_point(tp11)
        img.draw_point(tp0)

        img.img_show()
