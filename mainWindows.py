from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from welcome import Ui_Welcome
from mainwindow import Ui_MainWindow
from login import Ui_login

import pymysql

import os
import sys
import csv
import time
import inspect


import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale ,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
# from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

import pyqtgraph as pg

from GA import GA

class welcome(QWidget,Ui_Welcome):
    def __init__(self):
        super(welcome, self).__init__()
        self.setupUi(self)
        self.btn_1.clicked.connect(self.winSwitch1)
        self.btn_2.clicked.connect(self.winSwitch2)
        self.btn_3.clicked.connect(self.winSwitch3)
        self.btn_4.clicked.connect(self.winSwitch4)
        self.setWindowFlag(Qt.FramelessWindowHint)  # 隐藏框¡

    def winSwitch1(self):
        self.close()
        self.m = mainwindow()
        self.m.tabWidget.setCurrentIndex(0)

    def winSwitch2(self):
        self.close()
        self.m = mainwindow()
        self.m.tabWidget.setCurrentIndex(1)

    def winSwitch3(self):
        self.close()
        self.m = mainwindow()
        self.m.tabWidget.setCurrentIndex(2)

    def winSwitch4(self):
        self.close()
        self.m = mainwindow()
        self.m.tabWidget.setCurrentIndex(3)

class login(QWidget,Ui_login):
    def __init__(self):
        super(login, self).__init__()
        self.setupUi(self)
        self.initConnect()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

    def initConnect(self):
        self.btn_login.clicked.connect(self.on_login)


    def on_login(self):
        pass



class mainwindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(mainwindow, self).__init__()
        self.setupUi(self)
        self.show()
        self.initConnect()
        self.initUi()
        global prediction_value,loss_,y_test

    def initUi(self):
        self.showMaximized()
        # 表头
        # self.table_data.horizontalHeader().setStyleSheet('color:rgb(255,0,0)')
        # pyqtgraph
        pg.setConfigOption('background','#262721')
        pg.setConfigOption('foreground','#f0f0f0')
        # plt1
        self.p1 = QGridLayout(self.widget_plot_1)
        self.plt1 = pg.PlotWidget(title='Loss')
        self.canvas = self.plt1.plot(pen=(255,133,0),name='Loss',clear=True)
        self.plt1.setLabel(axis='left', text='Loss')
        self.plt1.setLabel(axis='bottom', text='Iter(hundred)')
        self.plt1.showGrid(x=True,y=True,alpha=0.5)
        # self.plt1.addLegend(size=(80, 50),offset=(-30,30))
        self.p1.addWidget(self.plt1)
        # plt2
        self.p2 = QGridLayout(self.widget_plot_2)
        self.plt2 = pg.PlotWidget(title='Predict VS Actual')
        self.plt2.setLabel(axis='left', text='Springback(%)')
        self.plt2.setLabel(axis='bottom', text='No.')
        self.plt2.showGrid(x=True, y=True, alpha=0.5)
        self.p2.addWidget(self.plt2)

    def initConnect(self):
        # action
        # terminate
        self.actionexit.triggered.connect(self.on_terminate)

        ## data base
        # login
        self.btn_login.released.connect(self.on_btn_login)
        # exit
        self.btn_exit.clicked.connect(self.on_btn_exit)
        # selected database
        self.cbb_database.currentIndexChanged.connect(self.on_database)
        # upload
        self.btn_upload.clicked.connect(self.on_btn_upload)
        # execute
        self.btn_execute.clicked.connect(self.on_btn_execute)
        # tab_1 open file
        self.btn_open.clicked.connect(self.on_btn_openFile)
        # insert row
        self.insertRow.triggered.connect(self.on_insertRow)
        # insert col
        self.insertCol.triggered.connect(self.on_insertCol)
        # delete row
        self.deleteRow.triggered.connect(self.on_deleteRow)
        # delete col
        self.deleteCol.triggered.connect(self.on_deleteCol)

        # NN
        #suspend
        self.btn_suspend.clicked.connect(self.on_suspend)
        #
        self.data = np.array([])
        # load data
        self.btn_load_data.clicked.connect(self.on_load_data)
        # load weight and biases
        self.btn_load_w_b.clicked.connect(self.on_load_w_b)
        # start train
        self.btn_startTrain.clicked.connect(self.NN_parameter)
        # delete Text
        self.editDelete.triggered.connect(self.on_deleteText)
        ## GA
        # load GA data
        self.btn_load_data_GA.clicked.connect(self.on_load_GA_data)
        # GA train
        self.btn_startTrain_GA.clicked.connect(self.on_train_GA)

        ## inference
        self.btn_browse_Search.clicked.connect(self.on_browse)
        self.btn_case_search.clicked.connect(self.on_case_search)
        self.btn_inference.clicked.connect(self.on_NN_Inference)
        self.btn_weight.clicked.connect(self.on_NN_Weights)

####################################################
#################### 工 具 栏 #######################
####################################################
    def on_terminate(self):
        try:
            # os._exit(0)
            sys.exit(0)
        except Exception as e:
            pass

####################################################
################## 数 据 管 理 ######################
####################################################
    def on_btn_login(self):

        user = self.userName.text()
        password = self.password.text()

        try:
            self.textEdit_database.append(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))  # 显示时间
            self.textEdit_database.append('<font color=\'#0000FF\'>\n正在连接...</font>')
            self.connect = pymysql.connect(user=user,password=password,database='expert_system',charset='utf8')
            QApplication.processEvents()  # 实时处理
            time.sleep(1)  # 延迟
            self.textEdit_database.append('<font color=\'#0000FF\'>登陆成功!</font>')
            self.btn_login.setEnabled(False)

        except Exception as e:
            self.textEdit_database.append('<font color=\'#FF0000\'>Error:用户名或密码输入有误！\n%s</font>' % e)
            QMessageBox.critical(self,'错误','用户名或密码输入有误！\n%s'%e)
            return 0
        # 游标
        self.cursor = self.connect.cursor()
        self.showDatabase()

    def on_database(self):
        self.database = self.cbb_database.currentText()
        try:
            if self.connect:
                self.showDatabase()

        except Exception as e:
            pass


    def showDatabase(self):
        self.table_data.clear()
        # 显示数据库
        if self.cbb_database.currentIndex() == 0:
            self.connect.rollback()
            return 0
        database = self.cbb_database.currentText()
        sql = ('select * from %s')%database
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        row = 0
        for line in result:
            col = 0
            for item in line:
                if type(item) != str:
                    newItem = QTableWidgetItem(str(item))
                else:
                    newItem = QTableWidgetItem(item)
                self.table_data.setItem(row,col,newItem)
                col += 1
            row += 1
        # 表结构
        self.tables_desc = self.cursor.description
        self.head = []
        for table in self.tables_desc:
            self.head.append(table[0])

        # 设置表的行列数
        if result:
            self.table_data.setRowCount(row)
            self.table_data.setColumnCount(col)
        else:
            self.table_data.setRowCount(1)
            self.table_data.setColumnCount(len(self.head))
        # 设置表头（字段）
        self.table_data.setHorizontalHeaderLabels(self.head)
        # self.textEdit_database.append(str(self.head))

    def on_btn_upload(self):
        self.textEdit_database.append(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))  # 显示时间
        self.textEdit_database.append('<font color=\'#0000FF\'>正在上传...</font>')
        # 清除数据库的数据
        sql = 'delete from %s'%self.database
        try:
            self.cursor.execute(sql)
            self.connect.commit()
        except Exception as e:
            QMessageBox.critical(self,'错误','%s'%e)

        # 插入
        row = self.table_data.rowCount()
        col = self.table_data.columnCount()
        for i in range(row):
            items = []
            for j in range(col):
                if self.table_data.item(i,j)!=None:# 判断数据是否为空
                    item =  self.table_data.item(i,j).text()
                    if item.isalpha() or item.isalnum():
                        items.append('"%s"'%item)
                    elif float(item):
                        items.append(item)
                    else:
                        QMessageBox.warning(self,'警告','%s行%s列数据错误'%(i,j))
                else:
                    items.append('null')
            ins = ','.join(items)

            sql_insert = 'insert into %s values('%self.database
            for i in range(col):
                if i<col-1:
                    sql_insert = sql_insert + items[i] + ','
                else:
                    sql_insert = sql_insert + items[i] + ')'

            try:
                self.cursor.execute(sql_insert)
                self.connect.commit()
            except Exception as e:
                QMessageBox.critical(self,'错误','%s'%e)
                self.textEdit_database(self,'%s'%e)
                self.connect.rollback()

        self.textEdit_database.append('<font color=\'#0000FF\'>上传成功!</font>')

    def on_btn_execute(self):
        sql_search = self.lineEdit_search.text()
        try:
            self.cursor.execute(sql_search)
            self.connect.commit()
            if sql_search.split()[0] == 'select':
                self.table_data.clear()
                result = self.cursor.fetchall()
                row = 0
                for line in result:
                    col = 0
                    for item in line:
                        if type(item) != str:
                            newItem = QTableWidgetItem(str(item))
                        else:
                            newItem = QTableWidgetItem(item)
                        self.table_data.setItem(row, col, newItem)
                        col += 1
                    row += 1
            else:
                self.showDatabase()
            self.textEdit_database.append('执行:%s'%sql_search)
            self.textEdit_database.append('<font color=\'#0000FF\'>执行成功!</font>')
        except Exception as e:
            self.textEdit_database.append('Error:%s'%e)
            QMessageBox.critical(self,'错误','%s'%e)
            self.connect.rollback()


    def on_btn_exit(self):
        try:
            self.cursor.close()
            self.connect.close()
            self.table_data.clear()
            self.textEdit_database.append('<font color=\'#0000FF\'>已退出!\n</font>')
            QMessageBox.information(self,'提示','已退出！')
            self.btn_login.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self,'错误','%s'%e)
            self.connect.rollback()

    def on_btn_openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  'open file',
                                                  './data',
                                                  '*.csv')
        if fileName == "":
            return

        self.filePath.setText(str(fileName))
        self.fileName = fileName
        self.loadCsv(fileName)

    def loadCsv(self,fileName):
        items = []
        rowCount = -1
        colCount = 0

        with open(fileName,'r') as f:
            for row in csv.reader(f):
                rowCount = rowCount+1
                colCount = len(row)
                for item in row:
                    items.append(item)
        self.table_data.setRowCount(rowCount)
        self.table_data.setColumnCount(colCount)
        self.table_data.setHorizontalHeaderLabels(items[:colCount])

        data = items[colCount:]
        for i in range(rowCount):
            for j in range(colCount):
                self.table_data.setItem(i,j,QTableWidgetItem(data[i*colCount+j]))

    def on_insertRow(self):
        curRow = self.table_data.rowCount()
        self.table_data.insertRow(curRow)

    def on_insertCol(self):
        curCol = self.table_data.columnCount()
        self.table_data.insertColumn(curCol)

    def on_deleteRow(self):
        curRow = self.table_data.rowCount()
        self.table_data.removeRow(curRow-1)

    def on_deleteCol(self):
        curCol = self.table_data.columnCount()
        self.table_data.removeColumn(curCol-1)


####################################################
################## 智 能 学 习 ######################
####################################################
    def on_suspend(self):
        try:
            # os.system("pause")
            print("suspend")

        except:
            pass

    def on_deleteText(self):
        if   self.tabWidget.currentIndex() == 0:
            self.textEdit_database.clear()
        elif self.tabWidget.currentIndex() == 1:
            self.textEdit_trainResult.clear()
        elif self.tabWidget.currentIndex() == 2:
            self.textEdit_inference.clear()
        elif self.tabWidget.currentIndex() == 3:
            pass
        else:
            pass

    def NN_parameter(self):
        try:
            net = []
            aFuncs = []
            # input layer 
            input_X = int(self.inputLayer.text())
            net.append(input_X)

            # hide layer
            h = int(self.hideLayer_0.text())
            aFunc_h = self.cbb_aFunc_0.currentIndex()
            net.append(h)
            aFuncs.append(aFunc_h)

            # output layer
            y = int(self.outputLayer.text())
            aFunc_y = self.cbb_aFunc_output.currentIndex()
            net.append(y)
            aFuncs.append(aFunc_y)

            # learning rate
            lr = float(self.learningRate.text())

            # 优化器
            optimizer = self.cbb_optimizer.currentIndex()

            # 损失函数
            lossFunc = self.cbb_lossFunc.currentIndex()

            # 训练周期
            trainEpoch = int(self.trainEpoch.text())

            # dropout
            dropout = float(self.dropout.text())

            # 模型保存
            saveModel = self.checkBox_saveModel.isChecked()
            saveImg = self.checkBox_saveImg.isChecked()

            # params
            parameters = {"网络结构:":net,
                          "激活函数:":aFuncs,
                          "学习率:":lr,
                          "优化器":optimizer,
                          "损失函数":lossFunc,
                          "训练周期:":trainEpoch,
                          "dropout:":dropout,
                          "保存模型:":saveModel,
                          "保存图片:":saveImg}
            param = ("<font color=\'#FFFFFF\'>网络结构:\t%s\n激活函数:\t%s\n学习率:\t%s\n优化器:\t%s"
                     "\n损失函数:\t%s\n训练周期:\t%s\ndropout:\t%s\n保存模型:\t%s\n保存图片:\t%s</font>"
                     %(net,aFuncs,lr,optimizer,lossFunc,trainEpoch,dropout,saveModel,saveImg))
            self.textEdit_trainResult.append('%s'%param)
            self.textEdit_trainResult.append('<font color=\'#e6db74\'>★</font>'*24)
            self.fit(net=net,aFuncs=aFuncs,lr=lr,optimizer=optimizer,epoch=trainEpoch,lossFunc=lossFunc)
        except Exception as e:
            self.textEdit_trainResult.append('<font color=\'#ff6ec7\'>Error : %s</font>' % e)
            QMessageBox.critical(self,'错误','%s'%e)


    def on_load_data(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'open file', './data', '*.csv')
        if fileName == "":
            return

        self.data_path.setText(str(fileName))

        # 读取数据
        df = pd.read_csv(str(fileName))
        # 显示数据摘要描述信息
        # print(df.describe())
        # print(df.columns)

        # 数据准备
        df = df.values
        # print('values:',df)
        df = np.array(df)
        # print('array:',df)

        inputLayer = df.shape[-1] - int(self.outputLayer.text())

        self.x_data = df[:, :inputLayer]
        self.y_data = df[:, inputLayer:]


        self.textEdit_trainResult.append('<font color=\'#ff6ec7\'>File Path : %s</font>'%str(fileName))
        self.textEdit_trainResult.append('samples : ' + str(df.shape[0]))
        self.inputLayer.setText(str(inputLayer))

    def on_load_w_b(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'open file', './Weights_and_Biases/optimal_w_b', '*.txt')
        if fileName == "":
            return
        self.w_b_path.setText(fileName)
        self.textEdit_trainResult.append('<font color=\'#ff6ec7\'>Weights and Biases Path : %s</font>'
                                         %str(fileName))

    def random_w_b(self,net):
        w1 = tf.Variable(tf.random_normal(shape=[net[0], net[1]]), name="weights")
        b1 = tf.Variable(tf.zeros(shape=[1, net[1]]) + 0.1, name="biases")
        w2 = tf.Variable(tf.random_normal([net[1], net[2]]), name="weights")
        b2 = tf.Variable(tf.zeros(shape=[1, net[-1]]) + 0.1, name="biases")
        params={
            'w1':w1,
            'w2':w2,
            'b1':b1,
            'b2':b2
        }
        return params

    def activation(self,A,activation_function=4):
        if activation_function == 4:
            return A
        else:
            with tf.name_scope("activation_function"):
                if activation_function == 0:
                    return tf.nn.sigmoid(A)
                elif activation_function == 1:
                    return tf.nn.relu(A)
                elif activation_function == 2:
                    return tf.nn.tanh(A)
                elif activation_function == 3:
                    return tf.nn.softmax(A)

    def optimizer(self,loss,lr,optimizer):
        if optimizer == 0:
            train_step = tf.train.AdamOptimizer(lr).minimize(loss)
        elif optimizer == 1:
            train_step = tf.train.AdagradOptimizer(lr).minimize(loss)
        elif optimizer == 2:
            train_step = tf.train.AdagradDAOptimizer(lr,epoch).minimize(loss)
        elif optimizer == 3:
            train_step = tf.train.FtrlOptimizer(lr).minimize(loss)
        elif optimizer == 4:
            train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        elif optimizer == 5:
            train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss)
        elif optimizer == 6:
            train_step = tf.train.ProximalAdagradOptimizer(lr).minimize(loss)
        elif optimizer == 7:
            train_step = tf.train.ProximalGradientDescentOptimizer(lr).minimize(loss)
        elif optimizer == 8:
            train_step = tf.train.RMSPropOptimizer(lr).minimize(loss)
        return train_step


    def fit(self, net, lr=0.01, aFuncs=None, optimizer=0, epoch=1000, lossFunc=0, keep_prob=1):
        #加载数据
        # boston = load_boston()
        global y_test

        test_rate = float(self.dsb_testRate.text())
        X_train,X_test,y_train,y_test = train_test_split(self.x_data,self.y_data,test_size=test_rate,random_state=0)
        # 归一化 [0,1]
        scaler_X_train = preprocessing.MinMaxScaler()
        X_train = scaler_X_train.fit_transform(X_train)
        scaler_X_test = preprocessing.MinMaxScaler()
        X_test = scaler_X_test.fit_transform(X_test)
        scaler_y_train = preprocessing.MinMaxScaler()
        y_train = scaler_y_train.fit_transform(y_train)
        scaler_y_test = preprocessing.MinMaxScaler()
        y_test_one = scaler_y_test.fit_transform(y_test)


        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default() as g:
            # 权值　偏置
            if self.w_b_path.text():
                # 　读取最佳训练权值/偏置
                w_b = open(self.w_b_path.text(),'r')
                params = eval(w_b.read())
            else:
                params = self.random_w_b(net)

            # model-神经网络结构
            with tf.name_scope('input_layer') as scope:
                global xs
                xs = tf.placeholder(shape=[None, net[0]], dtype=tf.float32, name="X")
                ys = tf.placeholder(shape=[None, net[-1]], dtype=tf.float32, name="Y")

            keep_prob_s = tf.placeholder(dtype=tf.float32)

            with tf.name_scope('weights1') as scope:
                w1 = tf.Variable(params['w1'],dtype=tf.float32)

            with tf.name_scope('biases1') as scope:
                b1 = tf.Variable(params['b1'],dtype=tf.float32)

            with tf.name_scope('weights2') as scope:
                w2 = tf.Variable(params['w2'],dtype=tf.float32)

            with tf.name_scope('biases2') as scope:
                b2 = tf.Variable(params['b2'],dtype=tf.float32)

            with tf.name_scope('hide_layer') as scope:
                h1 = self.activation(tf.matmul(xs, w1) + b1,activation_function=aFuncs[0])

            with tf.name_scope('OUT') as scope:
                global pred
                pred = self.activation(tf.matmul(h1, w2) + b2,activation_function=aFuncs[-1])

            with tf.name_scope("loss") as scope:
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred), reduction_indices=[1]))  # L2/mse均方误差
                tf.summary.scalar("loss", tensor=loss)

            with tf.name_scope('train_step') as scope:
                train_step = self.optimizer(loss,lr,optimizer)

            # trian
            init = tf.global_variables_initializer()  #初始化
            with tf.Session() as sess:
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter(logdir="log", graph=sess.graph)  #写tensorbord
                sess.run(init)
                loss_ = []
                for i in range(epoch + 1):
                    self.lcd_epoch.display(i)
                    QApplication.processEvents()  # 实时处理
                    time.sleep(0.01)  # 延迟

                    feed_dict_train = {xs: X_train, ys: y_train, keep_prob_s: keep_prob}
                    feed_dict_test = {xs: X_test, keep_prob_s: keep_prob}
                    loss_value, _, weight1,weight2= sess.run([loss, train_step,w1,w2], feed_dict=feed_dict_train)

                    if i % int(self.HS_LossStep.value()) == 0:
                        loss_.append(loss_value)
                        self.canvas.setData(loss_[1:])
                        self.plt1.plotItem.setTitle('Loss:%s' % loss_value)
                        #　测　试
                        prediction_value = sess.run(pred, feed_dict=feed_dict_test)

                        global real_pre
                        real_pre = scaler_y_test.inverse_transform(prediction_value)

                        # 评价
                        if self.cbb_lossFunc.currentIndex() == 0:
                            evaluate = loss_value
                        elif self.cbb_lossFunc.currentIndex() == 1:
                            evaluate = loss_value ** 0.5
                        elif self.cbb_lossFunc.currentIndex() == 2:
                            evaluate = np.sum(np.absolute(prediction_value - y_test)) / len(y_test)
                        elif self.cbb_lossFunc.currentIndex() == 3:
                            evaluate = r2_score(y_test, prediction_value)
                        else:
                            evaluate = None

                        status = '<font color=\'#FFFFFF\'> Epoch: [%.4d/%d], Loss:%g, evaluate:%.3f </font>' \
                                 % (i, epoch, loss_value, evaluate)
                        self.textEdit_trainResult.append(status)
                        self.pg_plot()

                        QApplication.processEvents()  # 实时处理
                        time.sleep(0.01)
                # 输出最后一次训练完成后的测试值
                # print('real pred:',type(real_pre))
                # print('y test:',type(y_test))
                #　计算平均绝对误差
                # print(((abs(real_pre-y_test)/y_test)).mean())
                # 保存最后一次训练的权重
                global weight_record
                weight_record = 'Weights_and_Biases/train_w_b/No_'+str(epoch)+'_train_weights.txt'
                weights = {'w1':weight1.tolist(),'w2':weight2.tolist()} #将np.array格式的w1 w2转换为list

                f = open(weight_record,'w')
                f.write(str(weights))
                f.close()
                self.textEdit_trainResult.append('<font color=\'#ffaa00\'>训练完毕！</font>')
                # 保存模型
                if self.checkBox_saveModel.isChecked():
                    # saver.save(sess=sess, save_path="./model/model.ckpt", global_step=epoch)  # 保存模型
                    saver.save(sess,'./model/model.ckpt',global_step=epoch)
                    self.textEdit_trainResult.append('<font color=\'#ffaa00\'>模型已保存！</font>')

                self.textEdit_trainResult.append('<font color=\'#e6db74\'>★</font>' * 24)
                # 验证
                # global prediction_value
                # prediction_value = sess.run(pred, feed_dict=feed_dict_test)
                # self.pg_plot()



    def pg_plot(self):
        global y_test,real_pre
        # self.plt2.plotItem.setTitle()
        self.plt2.addLegend(size=(80, 50), offset=(-30, 30))
        # self.plt2.plot(prediction_value[:,0],pen='r',symbol='star', symbolBrush=(237,177,32),name='pred',clear=True)
        # self.plt2.plot(y_test[:,0],pen='g',symbol='h',symbolBrush=(217,83,25),name='real')

        self.plt2.plot(real_pre[:, 0], pen='g',  name='pred', clear=True)
        self.plt2.plot(y_test[:, 0],pen=None,symbol='star',symbolBrush=(0,0,255),name='real')

    def on_load_GA_data(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'open file', './data', '*.csv')
        if fileName == "":
            return
        self.data_path_GA.setText(fileName)
        self.textEdit_trainResult.append('<font color=\'#ff6ec7\'>File Path : %s</font>'%fileName)

    def on_train_GA(self):
        try:
            dataPath = self.data_path_GA.text()
            net0 = int(self.GA_net0.text())
            net1 = int(self.GA_net1.text())
            net2 = int(self.GA_net2.text())
            net = [net0,net1,net2]
            NUM = net0*net1 + net1 + net1*net2 + net2
            GEN = int(self.GA_GEN.text())
            POP_SIZE = int(self.GA_POP_SIZE.text())
            CHROM = int(self.GA_CHROM.text())
            PC = float(self.GA_PC.text())
            PM = float(self.GA_PM.text())
            lr = float(self.GA_lr.text())
            epoch = int(self.GA_epoch.text())
            save_log = self.checkBox_save_log.isChecked()
            self.GA_NUM.setText(str(NUM))
            info = '<font color=\'#FFFFFF\'>网络结构：%s,优化参数：%s,遗传代数：%s,' \
                   '种群规模：%s,染色体长度：%s,交叉概率：%s,' \
                   '变异概率：%s,学习率：%s,训练周期：%s</font>' \
                   %(net,NUM,GEN,POP_SIZE,CHROM,PC,PM,lr,epoch)
            self.textEdit_trainResult.append(info)
            self.textEdit_trainResult.append('Evolving...')
            # GA 训练并返回权值/偏置文件保存位置
            w_b_file,logPath= GA(dataPath,net,lr,epoch,POP_SIZE,GEN,CHROM,NUM,PC,PM,save_log)
            self.textEdit_trainResult.append('Evolved!')
            self.textEdit_trainResult.append('<font color=\'#ffaa00\'>最优权值/偏置：：%s</font>'%w_b_file)
            if save_log==True:
                self.textEdit_trainResult.append('<font color=\'#ffaa00\'>详细信息请查看：%s</font>'%logPath)
        except Exception as e:
            self.textEdit_trainResult.append('<font color=\'#ff6ec7\'>Error : %s</font>' % e)
            QMessageBox.critical(self, '错误', '%s' % e)


    ####################################################
    ################## 专 家 推 理 ######################
    ####################################################
    def similarity(self,items, search_items,weights,interval):
        '''
        component_database:零件特征
        search_values：待匹配特征
        w：特征权重值
        interval:特征区间　
        '''
        # 　权重重新分配
        w_new = []
        if np.sum(weights) != 0:
            for i in weights:
                w_new.append(i/np.sum(weights))
            w = np.array(w_new)
        else:
            return 0
        x = np.array(items)
        y = np.array(search_items)
        i_ = np.array(interval)

        sim = np.sum((1-(np.abs(x-y))/i_)*w)
        return sim

    def add_tablewidget(self,row,i):
        for j in range(len(row)):
            print(row[j])
            self.tableWidget_output.setItem(i,j,QTableWidgetItem(str(row[j])))


    def weights(self,w1,w2):
        R = []
        S = []
        R_SUM = 0
        for i in range(w1.shape[0]):
            r_ij = 0
            for j in range(w2.shape[1]):
                for k in range(w1.shape[1]):
                    x = w2[k][j]
                    r_ij += w1[i][k] * (1 - np.exp(-x)) / (1 + np.exp(-x))

            y = r_ij
            R_ij = abs((1 - np.exp(-y)) / (1 + np.exp(-y)))
            R.append(R_ij)
            R_SUM += R_ij

        for i in range(w1.shape[0]):
            S_ij = round(R[i] / R_SUM,3) #保留三位有效数
            S.append(S_ij)
        return S

    def on_browse(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'open file', './data', '*.csv')
        if fileName == "":
            return
        self.file_Search.setText(str(fileName))
        df = pd.read_csv(str(fileName))
        col = len(list(df)) # 表头长度
        # 在原有表头上加一个相似度
        output_head = list(df) #　表头名称
        output_head.append('Similarity')
        self.tableWidget_input.setColumnCount(col)
        self.tableWidget_output.setColumnCount(col+1)
        self.tableWidget_input.setHorizontalHeaderLabels(list(df))
        self.tableWidget_output.setHorizontalHeaderLabels(output_head)

    def on_case_search(self):
        self.textEdit_inference.append(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))  # 显示时间
        self.textEdit_inference.append('<font color=\'#000000\'>正在检索...</font>')
        try:
            row = self.tableWidget_input.rowCount()
            col = self.tableWidget_input.columnCount()
            search_params = []
            index = []
            for i in range(col):
                try:
                    item = float(self.tableWidget_input.item(0,i).text())
                    search_params.append(item)
                    index.append(i)
                except:
                    pass

            file = self.file_Search.text()
            db = pd.read_csv(file)
            name = list(db) # 表头名
            global weight_coefficient
            cnt = 0
            sim_items = []
            for i in range(db.shape[0]):
                row_items = []
                interval = []
                w = []
                for j in index:
                    # 区间
                    max_j = db.loc[:, name[j]].max()  # j列最大值
                    min_j = db.loc[:, name[j]].min()  # j列最小值
                    interval_j = max_j - min_j
                    interval.append(interval_j)
                    search_j = float(self.tableWidget_input.item(0,j).text())
                    # 如果要检索的数值在检索的最小最大区间内,则进行正常相似度计算
                    # 否则认为数据库中待匹配的数值等于需检索的数值（search_j - row_items = 0)
                    # 此做法一：有利于相似度归一化，二:符合实际零件特征匹配需求
                    if min_j < search_j and search_j <max_j:
                        # 获取数据第i row ,j col
                        row_items.append(list(db[name[j]])[i])
                        # 　权重
                        w.append(weight_coefficient[j])
                    else:
                        w = np.zeros(shape=len(index),dtype=float).tolist()
                        break
                sim = self.similarity(row_items,search_params,w,interval)
                if sim>0.9:
                    row_i = list(db.loc[i,:])
                    row_i.append(round(sim*100,1))
                    sim_items.append(row_i)
                    cnt+=1
            self.tableWidget_output.clearContents()
            if cnt>5:
                self.tableWidget_output.setRowCount(cnt)
            else:
                self.tableWidget_output.setRowCount(5)
            for i in range(cnt):
                for j in range(col+1):
                    self.tableWidget_output.setItem(i,j,QTableWidgetItem(str(sim_items[i][j])))
            self.textEdit_inference.append('<font color=\'#000000\'>相似度大于90%%:</font>'
                                           '<font color=\'#238e23\'> %s 例</font>'%cnt)
        except Exception as e:
            self.textEdit_inference.append('<font color=\'#ff0000\'>Error : %s</font>' % e)
            QMessageBox.critical(self,'错误','%s'%e)


    def on_NN_Inference(self):
        try:
            # n = int(self.velocity.text())
            # # self.textEdit_inference.append('预测数据：%s'%n)
            n = 56
            x_test = self.x_data[n].reshape(-1,int(self.inputLayer.text()))
            # print('x_test:',x_test)
            y_real = self.y_data[n]
            # print('y_real:',y_real)
        except Exception as e:
            self.textEdit_inference.append('<font color=\'#ff0000\'>Error : %s</font>' % e)
            QMessageBox.critical(self,'错误','%s'%e)

        try:
            checkpoint_dir = './model/'
            # init = tf.global_variables_initializer() #加载已有模型不需要init
            with tf.Session(graph=tf.Graph()) as sess: # 在新的graph上run,避免和train的graph弄混
                # sess.run(init)
                model_file = tf.train.latest_checkpoint(checkpoint_dir)
                # 加载网络
                saver = tf.train.import_meta_graph(model_file + '.meta')
                # 加载参数
                saver.restore(sess, model_file)

                graph = tf.get_default_graph()
                # 记录参数名称
                # open('net_params.txt', 'w').write(str(graph.get_operations()))
                # 通过参数名称获取网络参数的定义
                xs = graph.get_tensor_by_name('input_layer/X:0')
                ys = graph.get_tensor_by_name('input_layer/Y:0')
                pred = graph.get_tensor_by_name('OUT/add:0')
                # 预测
                pred_value = sess.run(pred, feed_dict={xs:x_test})
                # print('pre_value:',pred_value)
                # print('real y:',y_real)

            # 零件类型预测
            prediction = 0
            k=0
            for i in pred_value:
                for j in range(len(i)):
                    if i[j]>i[k]:
                        k=j
                        prediction = j
            # print(prediction)
            if prediction == 0:
                pred_label = 'A柱上'
            elif prediction == 1:
                pred_label = 'A柱下'
            elif prediction  == 2:
                pred_label = 'B柱'
            else:
                pred_label = 'None'
            self.textEdit_inference.append(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))) #显示时间
            self.textEdit_inference.append('<font color=\'#000000\'>预测模型：%s</font>' % model_file)
            # self.textEdit_inference.append('<font color=\'#000000\'>真实值：%s</font>'%str(y_real))
            self.textEdit_inference.append('<font color=\'#238e23\'>推荐成形零件类型：%s</font>'%pred_label)
            self.textEdit_inference.append(' ')
        except Exception as e:
            self.textEdit_inference.append('<font color=\'#ff0000\'>Error : %s</font>' % e)
            QMessageBox.critical(self,'错误','%s'%e)

    def on_NN_Weights(self):
        try:
            plt.close()
            global weight_record
            weights = eval(open(weight_record,'r').read())
            w1 = weights['w1']
            w2 = weights['w2']
            #权重系数计算
            global weight_coefficient
            weight_coefficient = self.weights(np.array(w1),np.array(w2))

            self.textEdit_inference.append(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            self.textEdit_inference.append('<font color=\'#000000\'>权重系数：%s</font>' % weight_coefficient)
            ##权重系数直方图
            # weight_coefficient.sort()
            # 读取表头
            df = pd.read_csv(str(self.data_path.text()))
            label = []
            for i,v in enumerate(df.columns):
                label.append(v)
            label = label[:int(self.inputLayer.text())]

            weight_sequence = list(np.argsort(weight_coefficient))
            label_sequence = []
            for index,value in enumerate(weight_sequence):
                label_sequence.append(label[value])
            weight_coefficient.sort()
            plt.barh(label_sequence,weight_coefficient)
            plt.show()

        except Exception as e:
            self.textEdit_inference.append('<font color=\'#ff0000\'>Error : %s</font>' % e)
            QMessageBox.critical(self, '错误', '%s' % e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = welcome()
    w.show()
    sys.exit(app.exec_())