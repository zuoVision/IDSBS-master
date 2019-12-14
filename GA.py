'''
Created on 2017年11月21日

@author: ljs
'''
from ga_bp_binary.ga_encoding import ga_encoding
from ga_bp_binary.ga_decoding import ga_decoding,ga_decoding_individual
from ga_bp_binary.ga_calObject import ga_calObject
from ga_bp_binary.ga_calFitness import ga_calFitness
from ga_bp_binary.ga_selection import ga_selection
from ga_bp_binary.ga_crossover import ga_crossover
from ga_bp_binary.ga_mutation import ga_mutation
from ga_bp_binary.ga_replace import ga_replace
from ga_bp_binary.ga_getBest import ga_getBest
from ga_bp_binary.bp_object import bp_object
from ga_bp_binary.log import log
from ga_bp_binary.write_w_b import write_w_b
import numpy as np

import matplotlib.pyplot as plt


dataPath = 'data/A_part.csv'
net = [13,10,1]
lr = 0.01  #寻优时的学习率
epoch = 200 #寻优时的训练周期
POP_SIZE = 1#种群个体数量
GEN = 10#遗传迭代代数
CHROM = 8#染色体二进制   位数
NUM = net[0]*net[1]+net[1]+net[1]*net[2]+net[2] #11*10+10+10*1+1待优化权值与偏重数量
PC = 0.7#交叉概率
PM = 0.05#变异概率



def GA(dataPath,net,lr,epoch,POP_SIZE,GEN,CHROM,NUM,PC,PM,save_log=False):
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.ion()
    result = [[]]  # 存储最优解及其对应权值偏重
    Error = []
    pop = ga_encoding(POP_SIZE,CHROM,NUM)
    for i in range(1,GEN+1):
        x = ga_decoding(pop, CHROM, NUM)
        obj = ga_calObject(x,net,dataPath,lr,epoch)
        best_pop,best_fit = ga_calFitness(pop,obj)
        #如果这一代最优值高于上一代，就用上一代最优值代替这一代最差的
        if len(result) != 1 and best_fit>result[-1][0]:
            ga_replace(result[-1],pop,obj)
        result.append([best_fit,ga_decoding_individual(best_pop, CHROM, NUM),best_pop])
        #python中list,dict是可变对象，参数传递相当于引用传递，所以会改变变量本身，string,tuple,number是不可变对象
        ga_selection(pop,obj)
        ga_crossover(pop,PC)
        ga_mutation(pop, PM)
        Error.append(1/result[i][0])
        print('error:',1/result[i][0])
        plt.pause(0.001)
        try:
            ax.lines.remove(lines[0])
        except Exception as e:
            pass
        lines = ax.plot(np.arange(1,i+1),Error,c='b',lw=2)
    plt.ioff()
    plt.show()

    best = ga_getBest(result)
    file = write_w_b(best,net)

    # 保存实验数据
    logPath = ''
    if save_log==True:
        logPath = log(dataPath,net,lr,epoch,POP_SIZE,GEN,CHROM,NUM,PC,PM,result)

    return (file,logPath)
if __name__ == '__main__':
    w_b_file,logPath= GA(dataPath,net,lr,epoch,POP_SIZE,GEN,CHROM,NUM,PC,PM)
