'''
Created on 2017年11月23日

@author: ljs
'''
def ga_getBest(result):
    value = []
    for r in result[1:]:
        value.append(r[0])
    i = value.index(max(value))
    # print('最大适应度：',result[i+1][0])
    return result[i+1][1]
    
if __name__ == '__main__':
    pass