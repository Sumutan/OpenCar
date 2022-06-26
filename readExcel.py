import pandas as pd
import xlrd
import numpy as np
# tablePass = 'C:\\Users\\13390\\Desktop\\汽车座椅项目\\数据集\\lable.xlsx'

# 读取工作簿和工作簿中的工作表
# writer_1=pd.ExcelFile('C:\\Users\\13390\\Desktop\\汽车座椅项目\\数据集\\lable.xlsx')
# data_frame= writer_1.parse('Sheet1')
#
# print(data_frame)

def read_exceldata(usecols,tablePass='lengths.xlsx',sheet_name='Sheet1',print=False):
    data = pd.read_excel(tablePass, sheet_name=sheet_name, header=0,usecols=usecols)  # 想要读取usecols指定列的数据
    data.head()
    data.info()

    train_data = np.array(data)  # np.ndarray()
    excel_list = train_data.tolist()  # list

    # if print:
    return excel_list


if __name__ == '__main__':
    tablePass = 'lengths.xlsx'
    # data = pd.read_excel(tablePass, sheet_name='Sheet1', usecols=[0, 1, 2, 3, 4, 5])  # 想要读取第一列、第二列、和第三列的数据
    # data.head()
    # # print(data)
    #
    # train_data = np.array(data)  # np.ndarray()
    # excel_list = train_data.tolist()  # list
    excel_list=read_exceldata('B,K',tablePass)
    print("数据数量:" + str(len(excel_list)))
    print(excel_list)
    print(type(excel_list))
