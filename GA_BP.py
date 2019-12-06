import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import scale
import pandas as pd

######## GA
def fileWriter(path,var):
    var.sort()
    with open(path,'w+') as f:
        f.write('Gene\tparams\tlr\tepoch\tD_SIZE\tP_SIZE\tc_rate\tm_rate\tBOUND\tMFD\tmin_loss\n')
        for item in var:
            for i in item:
                f.write(str(i)+'\t')
            f.write('\n')
        f.close()
    print('数据已保存至：',path)

def load_csv(path):
    df = np.array(pd.read_csv(path))
    inputLayer = df.shape[-1]-1
    for i in range(df.shape[1]):
        df[:, i] = (df[:, i] - df[:, i].min()) / (df[:, i].max() - df[:, i].min())
    x_data = df[:,:inputLayer]
    y_data = df[:,inputLayer:]
    return (x_data,y_data)

def weight_biases(individual,net,DNA_SIZE,translate=True):
    w1 = individual[0:DNA_SIZE*(net[0]*net[1])].reshape(net[0]*net[1],DNA_SIZE)
    b1 = individual[DNA_SIZE*(net[0]*net[1]):DNA_SIZE*(net[0]*net[1]+net[1])].reshape(net[1],-1)
    w2 = individual[DNA_SIZE*(net[0]*net[1]+net[1]):DNA_SIZE*(net[0]*net[1]+net[1]+net[1]*net[2])].reshape(net[1],net[2]*DNA_SIZE)
    b2 = individual[DNA_SIZE*(net[0]*net[1]+net[1]+net[1]*net[2]):DNA_SIZE*(net[0]*net[1]+net[1]+net[1]*net[2]+net[2])].reshape(net[2],-1)
    if translate:
            w1 = np.array(translateDNA(w1).reshape(net[0],net[1]))
            b1 = np.array(translateDNA(b1).reshape(1,-1))
            w2 = np.array(translateDNA(w2).reshape(net[1],net[2]))
            b2 = np.array(translateDNA(b2).reshape(1,-1))

    params = {
            'w1':w1,
            'b1':b1,
            'w2':w2,
            'b2':b2
            }
    return params

# find non-zero fitness for selection
def get_fitness(loss): 
    return 1/loss


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop): 
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE*n_params).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


###### BP
def load_data():
    boston = load_boston()
    x_data,x_test,y_data,y_test = train_test_split(boston.data,boston.target,test_size=0.1,random_state=0)
    x_data = scale(x_data)
    x_test = scale(x_test)
    y_data = scale(y_data.reshape((-1,1)))
    y_test = scale(y_test.reshape((-1,1)))
    # print('x_data shape:',x_data.shape)
    # print("y_data shape:",y_data.shape)
    return (x_data,x_test,y_data,y_test)

def add_layer(inputs,w,b,activation=None):
    with tf.variable_scope("Weights"):
        Weights = tf.Variable(w, dtype=tf.float32,name="weights")
    with tf.variable_scope("biases"):
        biases = tf.Variable(b , dtype=tf.float32,name="biases")
    with tf.name_scope("A"):
        A = tf.add(tf.matmul(inputs, Weights), biases)

    if activation == 0:
        output = tf.nn.sigmoid(A)
    elif activation == 1:
        output = tf.nn.relu(A)
    elif activation == 2:
        output = tf.nn.tanh(A)
    else:
        output = A
    return output  

def fit(x_data,y_data,params,lr,epoch):
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    
    with tf.name_scope('input') as scope:
        x = tf.placeholder(shape=[None, net[0]], dtype=tf.float32, name="inputs")
        y = tf.placeholder(shape=[None, net[-1]], dtype=tf.float32, name="y_true")

    with tf.name_scope('hide_layer') as scope:
        h1 = add_layer(x,w1,b1,activation=1)

    with tf.name_scope('output') as  scope:
        output = add_layer(h1,w2,b2)

    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-output),reduction_indices=[1]))

    with tf.name_scope('train') as scope:
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            loss_value,_ = sess.run((loss,train_step),feed_dict = {x:x_data,y:y_data})
    return loss_value

################# GA_BP
def GA_BP(pop,x_data,y_data,lr=0.01,epoch=100,N_GENERATIONS=10):
    plt.ion()
    plt.title('pop size:%s/generations:%s/cross rate:%s/mutate rate:%s,bound:%s'%
        (POP_SIZE,N_GENERATIONS,CROSS_RATE,MUTATION_RATE,X_BOUND))
    records = []
    for i in range(1,N_GENERATIONS+1):
        F_values = []
        for individual in pop:
            params  = weight_biases(individual,net,DNA_SIZE)
            loss = fit(x_data,y_data,params,lr,epoch)  # compute function value by extracting DNA
            
            F_values.append(loss)

        plt.scatter(i,min(F_values),c='b',alpha=0.5);plt.pause(0.05)
        F_values  = np.array(F_values)

        records.append([i,n_params,lr,epoch,DNA_SIZE,POP_SIZE,CROSS_RATE,
            MUTATION_RATE,X_BOUND,np.argmax(F_values),min(F_values)])
        
        print("Generation %s\tpop_shape:%s\nMost fitted DNA index:%s\nMin Loss:%s\n"%
            (i,pop.shape,np.argmax(F_values),np.min(F_values)))    
        F_values = get_fitness(F_values)

        pop = select(pop, F_values)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child
    fileWriter(str(N_GENERATIONS)+'-Generations Recording'+'.txt',records)
    print('Done!')
    plt.ioff()
    plt.show()
    
################# 参数
net = [3,9,1]
n_params  = net[0]*net[1]+net[1]+net[1]*net[2]+net[2]
lr =  0.1
epoch = 1000

DNA_SIZE = 10            # DNA length
POP_SIZE = 40           # population size
CROSS_RATE = 0.7        # mating probability (DNA crossover)
MUTATION_RATE = 0.01    # mutation probability
N_GENERATIONS = 10
X_BOUND = [0, 1]         # x upper and lower bounds



#随机生成种群
pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*n_params))   # initialize the pop DNA
# print('pop shape:',pop.shape)

#加载样本数据
# x_data,x_test,y_data,y_test = load_data()
x_data,y_data = load_csv('filling_slirry_ratio_data.csv')
# print(x_data.shape,y_data.shape)

#GA_BP
GA_BP(pop,x_data,y_data,lr,epoch,N_GENERATIONS)
