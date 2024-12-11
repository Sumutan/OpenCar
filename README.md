# **Open-car**


### 项目来源与说明

汽车人体数据采集与分析系统前期研制，通过单目视觉与深度学习方法估计人体参数。

论文链接：[_A method of measuring human body size based on image processing_](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12473/2653539/A-method-of-measuring-human-body-size-based-on-image/10.1117/12.2653539.short)

### 运行环境

请参考[pytorch-openpose](https://github.com/Hzzone/pytorch-openpose)

由于模型文件较大，从仓库git clone 或者下载该项目后，需要将包含openpose网络模型的model文件下载解压放在项目根目录下

### 模型权重下载链接

链接：https://pan.baidu.com/s/1DpbFzKtUlXvOpdIVyMPuvg?pwd=sz7x  
提取码：sz7x

### 快速运行
```
python prediction.py
```

### 文件说明
```
image/	                        #存放训练用到的图像文件 
model/ & model_data/ & nets/    #分别存放openopse和deeplab的模型权重文件 
net/	                        #存放自己保存的模型Fusion monster          
output_processing.py    #处理openpose模型输出（candidate）以及读写excel所用到的相关函数 
lengths.xlsx            #训练模型用的数据集 
measure.py	        #该文件提供在图像中根据openopse给出的点位以及deeplab给出的边界绘制各种点位以及延长线的函数
picture_pre_process.py	#用与对原始图像做预处理的文件（根据名字分类等操作）
prediction.py	        #输入图片进行一次推理 
test.py			#用于测试语法打的草稿 可随意修改 
train_simple_modle.py	#将数据集（lenths.xlsx）读入后进行模型训练并评估 
writedownExcel.py       #负责实现excel的读写 
DataMaker.py            #批量制作训练数据集 
```

### 致谢
[pytorch-openpose](https://github.com/Hzzone/pytorch-openpose) <br>
[deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch) <br>
[Pytorch 搭建自己的DeeplabV3+语义分割平台](https://www.bilibili.com/video/BV173411q7xF/?spm_id_from=333.337.search-card.all.click&vd_source=1422dc9d01b5aaaaa4457804c50b68ae) <br>
