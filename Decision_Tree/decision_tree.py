from math import log
import operator
from matplotlib.font_manager import FontProperties
import  matplotlib.pyplot as plt
import pickle


#1.数据预处理
# 当熵中的概率由数据估计(特别是最大似然估计)得到时，所对应的熵称为经验熵(empirical entropy)

def createDataSet():
    """
    创建数据集和特征列表
    :return:
    """
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    features = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 分类特征
    return dataSet, features

def splitDataSet(dataSet, feature_index, value):
    """
    子集抽取：若索引为feature_index的特征取值为value，将满足条件的数据抽取出来组成一个子集
    :param dataSet:待划分的数据集
    :param feature_index:划分数据集的特征的索引,取值范围为[0,3]
    :param value:需要返回的特征的值,不同的特征取值范围有所不同
    :return:
    """
    retDataSet = []                                        #创建返回的数据集列表
    for featVec in dataSet:                             #遍历数据集
        if featVec[feature_index] == value:             #子集抽取：将某个特征满足指定取值的子集抽取出来
            #将featVec[feature_index]拿掉
            reducedFeatVec = featVec[:feature_index]
            reducedFeatVec.extend(featVec[feature_index + 1:])     #extend()附加列表
            retDataSet.append(reducedFeatVec)           #append()附加单个元素
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    计算信息增益，选择最优特征
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1                    #特征数量,减1是因为最后一列是label
    baseEntropy = calcShannonEnt(dataSet)                 #计算数据集的香农熵
    bestInfoGain = 0.0                                  #信息增益
    bestFeature = -1                                    #最优特征的索引值
    for i in range(numFeatures):                         #遍历所有特征
        #获取dataSet的第i个特征的取值向量
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                         #创建set集合{},元素不可重复,即获取每个特征的所有可能取值
        conditionalEntropy = 0.0                                  #经验条件熵
        for value in uniqueVals:                         #计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)         #满足特征i取值为value的子集，比如年龄=青年的子集，只不过这里都数字化了而已
            prob = len(subDataSet) / float(len(dataSet))           #计算子集占整个数据集的比例
            conditionalEntropy += prob * calcShannonEnt(subDataSet)     #根据公式计算经验条件熵
        infoGain = baseEntropy - conditionalEntropy                     #信息增益
        #print("第%d个特征的增益为%.3f" % (i, infoGain))            #打印每个特征的信息增益
        if (infoGain > bestInfoGain):                             #计算信息增益
            bestInfoGain = infoGain                             #更新信息增益，找到最大的信息增益
            bestFeature = i                                     #记录信息增益最大的特征的索引值
    return bestFeature                                             #返回信息增益最大的特征的索引值


def calcShannonEnt(dataSet):
    num_entries = len(dataSet)    #数据集的行数
    label_count = {}              #保存每个label出现次数的字典
    for featVect in dataSet:
        current_label = featVect[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1
    shannonEnt = 0.0
    for key in label_count:
        prob = float(label_count[key])/num_entries    #以频率来近似概率
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt


def majorityCnt(classList):
    """
    返回出现次数最多的分类标签
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #根据字典的值降序排列
    return sortedClassCount[0][0]        #返回出现次数最多的分类标签


#2.创建决策树
def createTree(dataSet, all_features, best_features=[], i=0):
    """
    创建决策树，并用字典来存储
    :param dataSet: 训练集
    :param all_features: 所有分类特征标签组成的列表，也就是所有的特征组成的列表；此处标签是指特征的名字：比如年龄、有无房子
    :param best_features:最优特征列表
    :param i:记录划分次数
    :return: 决策树，最优特征列表
    """
    i += 1
    classList = [example[-1] for example in dataSet]            #取分类标签(是否放贷:yes or no)
    #递归创建决策树时，递归有两个终止条件：第一个停止条件是所有的类标签完全相同，则直接返回该类标签；
    if classList.count(classList[0]) == len(classList):            #如果类别完全相同则停止继续划分
        return classList[0]
    #第二个停止条件是使用完了所有特征，仍然不能将数据划分仅包含唯一类别的分组，即决策树构建失败，特征不够用。
    # 此时说明数据纬度不够，由于第二个停止条件无法简单地返回唯一的类标签，这里挑选出现次数最多的类别作为返回值。
    if len(dataSet[0]) == 1:            #这标志着用完了所有特征：因为每次划分都会用掉一个特征，所有特征都遍历完时只剩标签（是否放贷）那一列
        return majorityCnt(classList)
    best_feat_index = chooseBestFeatureToSplit(dataSet)                #最优特征的索引
    bestFeatLabel = all_features[best_feat_index]                            #最优特征的标签,也就是最优特征的名字
    best_features.append(bestFeatLabel)
    # print('第%d次划分依据的最优特征为: %s' %(i, best_features[i-1]))     #打印每次划分选用的是哪个特征
    decision_tree = {bestFeatLabel:{}}                                    #根据最优特征的标签生成树
    del(all_features[best_feat_index])                                        #删除已经使用特征标签
    featValues = [example[best_feat_index] for example in dataSet]        #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                #去掉重复的属性值
    for value in uniqueVals:                                    #遍历特征，递归地创建决策树。
        copy_labels = all_features[:]
        decision_tree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, best_feat_index, value),
                                                         copy_labels, best_features, i)
    return decision_tree

#3可视化
def getNumLeafs(myTree):
    """
    获取决策树叶子节点的数目
    :param myTree:
    :return:
    """
    numLeafs = 0                                                #初始化叶结点数目
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])              #如果值为字典，则递归调用getNumLeafs(tree)
        else:   numLeafs +=1                                      #如果值不是字典，而是类标签，则该结点为叶节点
    return numLeafs

def getTreeDepth(myTree):
    """
    获取决策树的深度（层数）
    :param myTree:
    :return:
    """
    maxDepth = 0                                                #初始化决策树深度
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1                              #求决策树深度基本与求叶节点数目的思路是一样的，只需注意递归公式的差别
        if thisDepth > maxDepth: maxDepth = thisDepth      #更新层数
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制结点
    :param nodeTxt:节点名
    :param centerPt:文本位置
    :param parentPt:标注的箭头位置
    :param nodeType:节点格式
    :return:
    """
    arrow_args = dict(arrowstyle="<-")                                            #定义箭头格式
    font = FontProperties(fname=r"/usr/share/fonts/Chinese/simsun.ttc", size=14)        #设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',    #绘制结点
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
    """
    标注有向边的属性
    :param cntrPt:
    :param parentPt:
    :param txtString:标注内容
    :return:
    """
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]                                            #计算标注位置
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制决策树
    :param myTree:决策树（以字典形式存储）
    :param parentPt:标注内容
    :param nodeTxt:节点名
    :return:
    """
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                        #设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")                                            #设置叶结点格式
    numLeafs = getNumLeafs(myTree)                                                          #获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)                                                            #获取决策树层数
    firstStr = next(iter(myTree))                                                            #下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)                                                    #标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)                                        #绘制结点
    secondDict = myTree[firstStr]                                                            #下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD                                        #y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                                            #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))                                        #不是叶结点，递归调用继续绘制
        else:                                                                                #如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    """
    创建绘制面板
    :param inTree:
    :return:
    """
    fig = plt.figure(1, facecolor='white')                                                    #创建fig
    fig.clf()                                                                                #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                                #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))                                            #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))                                            #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                                #x偏移
    plotTree(inTree, (0.5,1.0), '')                                                            #绘制决策树
    plt.show()                                                                                 #显示绘制结果


#4.决策树用于分类

def tree_classify(inputTree, best_features, testVec):
    """
    使用决策树来进行分类
    :param inputTree:已经生成的决策树
    :param best_features: 最优特征标签列表
    :param testVec: 测试数据列表,顺序与最优特征对应
    :return: classLabel:分类结果
    """
    firstStr = next(iter(inputTree))
    second_dict = inputTree[firstStr]
    feat_index = best_features.index(firstStr)
    for key in second_dict.keys():
        if testVec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                classLabel = tree_classify(second_dict[key], best_features, testVec)
            else:
                classLabel = second_dict[key]
    return classLabel


#5.决策树的存储与读取
def storeTree(inputTree, filename):
    """
    存储决策树
    :param inputTree: 已经生成的决策树
    :param filename: 决策树存储的文件名
    :return:
    """
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)

def grabTree(filename):
    """
    从文件读取决策树
    :param filename: 存储决策树的文件名
    :return: 决策树的字典
    """
    fr = open(filename, 'rb')
    return pickle.load(fr)


#6.使用自写的决策树分类器来预测隐形眼镜的类型  #7.后面会在另一个模块中利用sklearn中的决策树做同样的事

def load_file(filename):
    fr = open(filename)
    lenses_data = [line.strip().split('\t') for line in fr.readlines()]
    features = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses_data, features

if __name__ == '__main__':
    dataSet, all_features = createDataSet()
    #print(dataSet)
    # print(calcShannonEnt(dataSet))   #在该数据集下的经验熵为0.9709505944546686
    #print('最优特征索引值：'+str(chooseBestFeatureToSplit(dataSet))) #信息增益最大的特征，即最优特征索引为：2，也就是房子！
    #best_features = []  #初始化最优特征列表
    #i = 0            #初始化划分次数
    # 因为best_features始终指向的是同一段内存空间,在调用createTree()生成决策树的过程中，它会逐个将选取的最优特征加入到此列表,
    # 后面只要使用这个变量名，就仍可以使用该列表
    #lending_tree = createTree(dataSet, all_features, best_features)
    # print(best_features)  # ['有自己的房子', '有工作'],成功记录了最优特征
    # #print('需要%d次划分' %(i))  #试图直接通过i得到划分次数，但结果是：需要0次划分，因为i是int型，不是引用类型变量
    # #so,只能通过best_features的长度来计算划分次数了
    # print('划分次数： %d' %(len(best_features)))
    # print(lending_tree)
    # # createPlot(lending_tree)
    # testVec = [0, 1]            #表示没有房子，但有工作
    # classify_result = tree_classify(lending_tree, best_features, testVec)
    # if classify_result == 'yes':
    #     print('放贷')
    # if classify_result == 'no':
    #     print('不放贷')
    #storeTree(lending_tree, 'lending_decision_tree_file.txt') #存成2进制文件,不过直接打开是乱码，博客中是十六进制模式显示的结果
    # myTree = grabTree('lending_decision_tree_file.txt')
    # print(myTree)    #{'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
    lenses_data, features = load_file('lenses.txt')
    lenses_best_features = []
    lenses_tree = createTree(lenses_data, features, lenses_best_features)
    print(lenses_best_features)
    print('划分次数： %d'%(len(lenses_best_features)))
    print(lenses_tree)
    createPlot(lenses_tree)

