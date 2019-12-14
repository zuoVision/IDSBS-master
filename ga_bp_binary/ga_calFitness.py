
def ga_calFitness(pop,obj):
    best_pop = pop[0]
    best_fit = obj[0]
    for i in range(1,len(pop)):
        if(obj[i]>best_fit):
            best_fit = obj[i]
            best_pop = pop[i]
    #best_pop最优个体[1,0,0,1,0...],best_fit最优适应度,
    return [best_pop,best_fit]

if __name__ == '__main__':
    pass
