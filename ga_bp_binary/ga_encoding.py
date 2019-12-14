'''
Created on 2017年11月21日

@author: ljs
'''
import random
import numpy as np

def ga_encoding(pop_size,chrom,num):
    # pop = [[]]
    # for i in range(pop_size):
    #     temp = []
    #     for j in range(chrom*num):
    #         temp.append(random.randint(0,1))
    #     pop.append(temp)
    # print(np.array(pop).shape)
    # print(np.array(pop[1:]).shape)
    pop = np.random.randint(2, size=(pop_size, chrom * num))
    return pop
    
if __name__ == '__main__':
    pass
