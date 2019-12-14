import os
import numpy as np

def write_w_b(best,net):
    cwd = os.getcwd() + '/Weights_and_Biases/'
    name = 'most_fitted_weights_and_biases.txt'
    if not os.path.exists(cwd):
        os.mkdir(cwd)
    file = cwd + name
    with open(file,'w') as f:
        w1 = np.array(best[: net[0] * net[1]]).reshape([net[0], net[1]]).tolist()
        w2 = np.array(best[net[0] * net[1]: net[0] * net[1]+net[1]*net[2]]).reshape([net[1], net[2]]).tolist()
        b1 = np.array(best[net[0] * net[1]+net[1]*net[2]:-net[2]]).reshape([1, net[1]]).tolist()
        b2 = np.array(best[-net[2]:]).reshape([1, net[2]]).tolist()

        params = {'w1':w1,
                  'w2':w2,
                  'b1':b1,
                  'b2':b2}
        f.write(str(params))
        f.close()
    # print('权值/偏置已保存至：',file)
    return file