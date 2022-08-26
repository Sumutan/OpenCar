import os
import re
import shutil

"""
该文件用于更具动作标号对图片进行分类存放
"""

"""根据输入的路径创建文件夹"""
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("build a new folder:" + path)
    else:
        print("---  The folder path %s is existed!  ---"%path)


if __name__ == '__main__':
    for i in range(1, 6):
        mkdir("../images/dataset_raw/action" + str(i))  # 调用函数创建五个动作的分类文件夹

    path = "../images/dataset_raw"

    # 获取该目录下所有文件，存入列表中
    fileList = os.listdir(path)

    n = 0
    for i in fileList:
        oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符
        reg = r'\d+'
        matchObj = re.findall(reg, fileList[n])
        print(fileList[n])
        print(n)
        Name = {"id": matchObj[0], "action": matchObj[1]}
        newname = path + os.sep + "action" + matchObj[1] + os.sep + fileList[n]
        shutil.copyfile(oldname, newname)
        n += 1


