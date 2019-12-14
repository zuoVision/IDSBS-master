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

dataPath = 'A_part.csv'
net = [13,10,1]
lr = 0.01  #寻优时的学习率
epoch = 200 #寻优时的训练周期
POP_SIZE = 1#种群个体数量
GEN = 1#遗传迭代代数
CHROM = 8#染色体二进制   位数
NUM = net[0]*net[1]+net[1]+net[1]*net[2]+net[2] #11*10+10+10*1+1待优化权值与偏重数量
PC = 0.7#交叉概率
PM = 0.05#变异概率
result = [[]]#存储最优解及其对应权值偏重
real_lr = 0.01 #使用最佳参数训练时的学习率
real_epoch = 1000 #使用最佳参数训练时的训练周期

def GA(dataPath,net,lr,epoch,POP_SIZE,GEN,CHROM,NUM,PC,PM,result,real_lr,real_epoch):
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
        print('GEN %s over !'%i)
    print('result:',result)
    # for r in result:
    #     print('r:',r)
    best = ga_getBest(result)
    write_w_b(best,net)
    print('最佳训练参数:',best)
    # 用最优权值偏置训练神经网络　
    # bp_object(best,net,dataPath,real_lr,real_epoch)
    # 保存实验数据
    log(dataPath,net,lr,epoch,GEN,CHROM,NUM,PC,PM,real_lr,real_epoch,result)

if __name__ == '__main__':
    GA(dataPath,net,lr,epoch,POP_SIZE,GEN,CHROM,NUM,PC,PM,result,real_lr,real_epoch)
