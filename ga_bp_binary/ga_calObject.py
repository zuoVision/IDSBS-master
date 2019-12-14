
from ga_bp_binary.bp_train import bp_train 

def ga_calObject(x,net,dataPath,lr,epoch):
    obj = []
    #x是整个种群，xi是每个个体
    for xi in x:
        temp = bp_train(xi,net,dataPath=dataPath,lr=lr,epoch=epoch)
        obj.append(temp)
    return obj
  
if __name__ == '__main__':
    pass