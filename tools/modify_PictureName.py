import os
import re


'''
批量修改文件名，在不足2位数字的文件前补0，如1(2).jpg->01(2).jpg
'''
# path = input('请输入文件路径(结尾加上/)：')
path = "../images/img_processed"

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)

n = 0
for i in fileList:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符

    reg=r'\d+'
    matchObj = re.findall(reg, fileList[n])
    Name={"id":matchObj[0],"action":matchObj[1]}

    # 设置新文件名
    if len(Name["id"])<2:
        newname = path + os.sep + '0' + fileList[n]
        print(oldname, '======>', newname)
        os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名

    n += 1




