# **Open-car**



## 项目来源与说明

汽车座椅自适应系统研制&汽车人体数据采集与分析系统

该项目主要完成以上两个课题的视觉测量部分

主要通过视觉测量图形中像素点间的距离来推断实物长度



特征点提取部分的模型来自：https://github.com/Hzzone/pytorch-openpose

语义风格部分的模型来自：https://github.com/bubbliiiing/deeplabv3-plus-pytorch

该项目主要对上述两个项目进行了整合，然后根据我们的项目需求进行代码上的补充。

更多参考资料：https://www.bilibili.com/video/BV173411q7xF?spm_id_from=333.337.search-card.all.click&vd_source=1422dc9d01b5aaaaa4457804c50b68ae



## 运行环境

与openpose项目相同，请参考openpose项目中的环境配置说明

由于模型文件较大，从仓库git clone 或者下载该项目后，需要将包含openpose网络模型的model文件下载解压放在项目根目录下

### model下载链接

链接：https://pan.baidu.com/s/1DpbFzKtUlXvOpdIVyMPuvg?pwd=sz7x 
提取码：sz7x

### 下载与上传修改
git clone https://github.com/Sumutan/OpenCar 快速拷贝到本地
提交项目修改：
git remote add upstream https://github.com/Sumutan/OpenCar   （添加提交地址，仅第一次提交前运行一次）

git fetch upstream 下载最新版本代码
git merge upstream/main 把远程最新代码合并到自己的代码中
然后可以提交三连：
git add .     添加项目文件（. 为所有文件，也可以是某个具体文件名字）
git commit -m "提交说明"    本地提交
git push -u origin main        网络提交
然后三连提交修改就完成了(本机测试有效)


### 文件说明
Fusion monster 主函数








