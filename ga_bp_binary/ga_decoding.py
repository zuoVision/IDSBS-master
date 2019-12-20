'''
Created on 2017年11月21日

@author: ljs
'''
import math
import pandas as pd
from ga_bp_binary.ga_encoding import ga_encoding


def ga_decoding(pop,chrom,num,dataPath):

    population = [[]]
    for p in pop:
        temp = []
        for i in range(num):
            t = 0
            for j in range(chrom):
                t += p[j + chrom * i] * (math.pow(2, j))
            temp.append(t)
        population.append(temp)
        print('population', population)
    return reduce(population[1:], chrom,dataPath)

def reduce(pop,chrom,dataPath):
    # load data
    df = pd.read_csv(dataPath)
    # min  max value
    df_min = list(df.min())
    df_max = list(df.max())
    pop_std = [[]]
    pop_norm = [[]]
    i=0
    for po in pop:

        standardization = []
        normalization = []
        for p in po:
            p = (p/math.pow(2,chrom))*(df_max[i]-df_min[i])+df_min[i]
            standardization.append(round(p,8))
            p = normalized(p, df_max[i], df_min[i])
            normalization.append(round(p, 8))
            i+=1
        pop_std.append(standardization)
        pop_norm.append(normalization)
    return pop_std[1:],pop_norm[1:]

def normalized(v,max,min):
    return (v-min)/(max-min)

def ga_decoding_individual(individual,chrom,num):
    temp = []
    for i in range(num):
        t = 0
        for j in range(chrom):
            t += individual[j+chrom*i] * (math.pow(2, j))
        temp.append(t)
    result = []
    for t in temp:
        t = -0.5 + t*(1/255)
        result.append(round(t,8))
    return result    
        
if __name__ == '__main__':
    dataPath = 'A_part.csv'
    pop = ga_encoding(1,10,13)
    std,norm = ga_decoding(pop,10,13,dataPath)
    print(std)
    print(norm)