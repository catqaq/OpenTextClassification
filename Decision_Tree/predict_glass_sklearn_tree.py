#Sklearn之使用决策树预测隐形眼镜类型
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus


if __name__ == '__main__':
    # fr = open('lenses.txt')
    # lenses_data = [inst.strip().split('\t') for inst in fr.readlines()]
    # print(lenses_data)
    # lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # dtc = tree.DecisionTreeClassifier()
    # lenses_data = dtc.fit(lenses_data, lensesLabels)  #调用类DecisionTreeClassifier中的方法fit()来生成一个决策树分类器
    #ValueError: could not convert string to float: 'young'
    # 因为在fit()函数不能接收string类型的数据，通过打印的信息可以看到，数据都是string类型的。
    # 在使用fit()函数之前，我们需要对数据集进行编码，这里可以使用两种方法：
    #     LabelEncoder：将字符串转换为增量值
    #     OneHotEncoder：使用One-of-K算法将字符串转换为整数
    # 为了对string类型的数据序列化，需要先生成pandas数据，这样方便我们的序列化工作。
    # 这里我使用的方法是，原始数据->字典->pandas数据，编写代码如下：
    with open('lenses.txt', 'r') as fr:
        lenses_data = [line.strip().split('\t') for line in fr.readlines()]
    lenses_target = []
    for each in lenses_data:
        lenses_target.append(each[-1])        #提取每组数据的最后一列，也就是类别，保存在列表里

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate'] #特征列表
    lenses_list = []                          #保存每个特征取值的临时列表
    lenses_dict = {}                          #保存lenses数据的字典，用于生成pandas
    for each_label in lensesLabels:
        for each in lenses_data:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list  #将 特征-对应的取值列表 对加入字典
        lenses_list = []                      #将lenses_list“归零”，很关键
    print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)    #生成pandas.DataFrame
    #print(lenses_pd)     #打印pandas数据
    #下面将数据序列化,也就是数字化
    le = LabelEncoder()  # 创建LabelEncoder()对象，用于序列化
    for col in lenses_pd.columns:  # 为每一列序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])  #fit_transform函数Fit label encoder and return encoded labels
    #print(lenses_pd)
    clf = tree.DecisionTreeClassifier(max_depth=4)  # 创建DecisionTreeClassifier()类的对象
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)  # 使用数据，构建决策树
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,  # 绘制决策树
                         feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("lenses_tree.pdf")  # 保存绘制好的决策树，以PDF的形式存储。
    print(clf.predict([[1, 1, 1, 0]])) #预测，['hard']






