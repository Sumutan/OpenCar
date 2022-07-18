"""
用与对原始图像做预处理的文件（根据名字分类等操作）
在未来不一定使用
"""
import os
import matplotlib.pyplot as plt
import copy
from src import util
from src.body import Body
from demo_functions import candidateright
import time
import cv2
import numpy as np
from measure import Img


def extend(p1, p2, k=1):
    p3 = (int((p1[0] - (p2[0] - p1[0]) / k)), int((p1[1] - (p2[1] - p1[1]) / k)))
    p4 = (int((p2[0] + (p2[0] - p1[0]) / k)), int((p2[1] + (p2[1] - p1[1]) / k)))
    return p3, p4


candidate_picture_files = ["", r"images/dataset/action1/", r"images/dataset/action2/",
                           r"images/dataset/action3/", r"images/dataset/action4/",
                           r"images/dataset/action5/"]
# save_pathes = ["", 'images/img_processed/action1_processed/', 'images/img_processed/action2_processed/',
#                'images/img_processed/action3_processed/',
#                'images/img_processed/action4_processed/','images/img_processed/action5_processed/']

"""根据动作分组选择路径"""
action = 1  # 1-5为图片，6-10为动作1-5对应的测试模式（不保存）
candidate_picture_file = candidate_picture_files[action]
# save_path = save_pathes[action]

# excel_path = r"C:\Users\10355\Desktop"

global r_image

if __name__ == "__main__":
    time_start = time.time()
    body_estimation = Body('model/body_pose_model.pth')  # 生成openpose模型

    """批量处理模式"""
    # photopath = os.listdir(candidate_picture_file)  # 绑定输入照片文件夹路径
    # num_picture = len(photopath)
    # print(photopath)
    # excelpath = './'
    # npicture = 1
    #
    # for i in range(num_picture):
    #     test_image = photopath[i]
    #     print(candidate_picture_file + test_image)
    #
    #     oriImg = cv2.imread(candidate_picture_file + test_image)  # B,G,R order
    #     candidate, subset = body_estimation(oriImg)  # 图片推理，
    #     canvas = copy.deepcopy(oriImg)
    #     canvas = util.draw_bodypose(canvas, candidate, subset)
    #
    #     # print(candidate)
    #     # print(subset)
    #     # print(np.shape(candidate))
    #     # candidateright(candidate, subset)
    #
    #     if action == 1:
    #         "P1绘制肩宽延长线"
    #         plt.imsave(save_path + test_image, canvas[:, :, [2, 1, 0]])
    #         img = Img(save_path + test_image, type='color')
    #         p1 = candidate[2][0], candidate[2][1]
    #         p2 = candidate[5][0], candidate[5][1]
    #         p3, p4 = extend(p1, p2)
    #         img.draw_line(p3, p4, r=1, color=(0, 0, 255))
    #         # img.img_show()
    #         img.saveimg(save_path + test_image)
    #
    #     elif action == 2:
    #         "P2绘制肩宽延长线"
    #         plt.imsave(save_path + test_image, canvas[:, :, [2, 1, 0]])
    #         img = Img(save_path + test_image, type='color')
    #         p1 = candidate[8][0], candidate[8][1]
    #         p2 = candidate[11][0], candidate[11][1]
    #         p3, p4 = extend(p1, p2)
    #         img.draw_line(p3, p4, r=1, color=(0, 0, 255))
    #         # img.img_show()
    #         img.saveimg(save_path + test_image)
    #
    #     elif action == 3:
    #         "P3绘制肩宽延长线"
    #         plt.imsave(save_path + test_image, canvas[:, :, [2, 1, 0]])
    #         img = Img(save_path + test_image, type='color')
    #         p2 = candidate[11][0], candidate[11][1]
    #         p3 = int(p2[0] - 50), int(p2[1])
    #         p4 = int(p2[0] + 50), int(p2[1])
    #         img.draw_line(p3, p4, r=1, color=(0, 0, 255))
    #         # img.img_show()
    #         img.saveimg(save_path + test_image)
    #
    #     elif action == 5:
    #         "P5绘制肩宽延长线"
    #         plt.imsave(save_path + test_image, canvas[:, :, [2, 1, 0]])
    #
    #     else:
    #         print("error:action out of range")
    #         break

    photofile=candidate_picture_file
    photoName="1(1).jpg"


    test_image_path = photofile+photoName
    print(test_image_path)
    print(os.path.exists(test_image_path))

    oriImg = cv2.imread(test_image_path)  # B,G,R order
    candidate, subset = body_estimation(oriImg)  # 图片推理，
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    print(candidate)
    print(subset)
    print(np.shape(candidate))
    candidateright(candidate, subset)

