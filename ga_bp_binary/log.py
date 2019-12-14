import time
import os
import numpy as np
def log(dataPath=None,net=None,lr=None,epoch=None,POP_SIZE=None,
        GEN=None,CHROM=None,NUM=None,PC=None,PM=None,result=None):
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filePath = str(os.getcwd()) + '/log/GA'
    if not os.path.exists(filePath):
        os.mkdir(filePath)
    filePath = filePath + '/event-GA-BP.txt'
    with open(filePath,'a') as f:
        event = '*'*50 + '\n' + \
                '* 保存时间：\t【' + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '】\n' \
                '* 样本路径：\t' + dataPath + '\n' + \
                '* 网络结构：\t' + str(net) +  '\n'+ \
                '* 学习率：\t' + str(lr) + '\n' + \
                '* 训练周期：\t' + str(epoch) + '\n' + \
                '* 种群规模：\t' + str(POP_SIZE) + '\n' + \
                '* 遗传代数：\t' + str(GEN) + '\n' + \
                '* 染色体长度：\t' + str(CHROM) + '\n' + \
                '* 待优化参数数目：\t' + str(NUM) + '\n' + \
                '* 交叉概率：\t' + str(PC) + '\n' + \
                '* 变异概率：\t' + str(PM) + '\n' + \
                '*'*50  + '\n'
        f.write(event)
        for i in result:
            i = np.array(i)
            if i.shape[0] != 0:
                log = '最佳适应度：\t' + str(i[0])  + '\n' \
                      '最优权值/偏置：\t' + str(i[1]) + '\n'\
                      '最优个体：\t' +  str(i[2]) + '\n' +  ' '
                f.write(log)
        f.close()
    # print('数据已保存至：',filePath)
    return filePath

