
from ga_bp_binary.bp_train import bp_train 
import time
def ga_calObject(x,net,dataPath,lr,epoch):
    obj = []
    #x是整个种群，xi是每个个体
    # start = time.time()
    for xi in x:
        temp = bp_train(xi,net,dataPath=dataPath,lr=lr,epoch=epoch)
        obj.append(temp)
    # end = time.time() - start
    # print('train time:%s:%s:%s' % (int(end // 3600), int(end // 60), int(end % 60)))
    return obj
  
if __name__ == '__main__':
    pass