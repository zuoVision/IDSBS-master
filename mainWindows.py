from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from welcome import Ui_Welcome
from mainwindow import Ui_MainWindow
from login import Ui_login

import pymysql

# import winsound
import os
import sys
import csv
import time
import inspect
from BP import *

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale ,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

import pyqtgraph as pg




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
        self.plt2.setLabel(axis='left', text='Springback')
        self.plt2.setLabel(axis='bottom', text='No.')
        self.plt2.showGrid(x=True, y=True, alpha=0.5)
        self.p2.addWidget(self.plt2)

        # hide layer
        self.h = 0
        self.hideLayers = []
        self.cbb_aFuncs = []
        self.btn_subtract.setEnabled(False)
        self.vLayout = QVBoxLayout(self.widget_hideLayer)

    def initConnect(self):
        # action
        # terminate
        self.actionexit.triggered.connect(self.on_terminate)

        # data
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
        # start train
        self.btn_startTrain.clicked.connect(self.NN_parameter)
        # delete Text
        self.editDelete.triggered.connect(self.on_deleteText)
        # add hide layer
        self.btn_add.clicked.connect(self.on_addHideLayer)
        # subtract hide layer
        self.btn_subtract.clicked.connect(self.on_subtractHideLayer)

        # inference
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
        self.textEdit_database.append('<font color=\'#0000FF\'>上传成功!</font>')
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
                if self.table_data.item(i,j).text():# 判断数据是否为空
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

    def on_addHideLayer(self):
        self.h += 1
        # line edit
        self.hideLayers.append("self.hideLayer_%s"%self.h)
        self.hideLayers[-1] = QLineEdit()
        self.hideLayers[-1].setFixedSize(80,23)
        self.hideLayers[-1].setClearButtonEnabled(True)

        # cbbox
        self.cbb_aFuncs.append('self.cbb_aFunc_%s'%self.h)
        self.cbb_aFuncs[-1] = QComboBox()
        self.cbb_aFuncs[-1].setFixedSize(80,21)
        self.cbb_aFuncs[-1].addItems(['sigmoid','relu','tanh','None'])

        # layout
        self.vLayout.addWidget(self.hideLayers[-1])
        self.vLayout.addWidget(self.cbb_aFuncs[-1])

        self.btn_subtract.setEnabled(True)

    def on_subtractHideLayer(self):
        self.hideLayers[-1].close()
        self.hideLayers.pop(-1)
        self.cbb_aFuncs[-1].close()
        self.cbb_aFuncs.pop(-1)
        if len(self.hideLayers) == 0 or len(self.cbb_aFuncs) == 0:
            self.btn_subtract.setEnabled(False)

    def NN_parameter(self):
        try:
            net = []
            aFuncs = []
            # input layer 
            input_X = int(self.inputLayer.text())
            net.append(input_X)

            # hide layer
            h_0 = int(self.hideLayer_0.text())
            aFunc_0 = self.cbb_aFunc_0.currentIndex()
            net.append(h_0)
            aFuncs.append(aFunc_0)

            hide_names = []
            hide_aFuncs = []
            for i in range(len(self.hideLayers)):
                hide_names.append('h_%s'%(i+1))
                hide_aFuncs.append('aFuncs_%s'%(i+1))
                hide_names[i] = int(self.hideLayers[i].text())
                hide_aFuncs[i] = self.cbb_aFuncs[i].currentIndex()
                net.append(hide_names[i])
                aFuncs.append(hide_aFuncs[i])

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
        if self.data_path.text().endswith('iris.csv'):
            self.outputLayer.setText('3')
        # 数据归一化
        for i in range(df.shape[1]):
            df[:, i] = (df[:, i] - df[:, i].min()) / (df[:, i].max() - df[:, i].min())

        self.x_data = df[:, :inputLayer]
        self.y_data = df[:, inputLayer:]


        self.textEdit_trainResult.append('<font color=\'#ff6ec7\'>File Path : %s</font>'%str(fileName))
        self.textEdit_trainResult.append('samples : ' + str(df.shape[0]))
        self.inputLayer.setText(str(inputLayer))


    def iris_data(self,CSV_FILE_PATH):
        data = np.array(pd.read_csv(CSV_FILE_PATH))
        features = data[:, 0:4]
        rows = data.shape[0]
        labels = np.array(np.zeros([rows, 3]))  #
        for i in range(rows):
            label = int(data[i][4])
            labels[i][label] = 1
        return features, labels

    def add_layer(self,inputs, input_size, output_size, activation_function=None):
        with tf.variable_scope("Weights"):
            Weights = tf.Variable(tf.random_normal(shape=[input_size, output_size]), name="weights")
        with tf.variable_scope("biases"):
            biases = tf.Variable(tf.zeros(shape=[1, output_size]) + 0.1, name="biases")
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        # with tf.name_scope("dropout"):
        #     Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=keep_prob_s)
        if activation_function == 4:
            return Wx_plus_b
        else:
            with tf.name_scope("activation_function"):
                if activation_function == 0:
                    return tf.nn.sigmoid(Wx_plus_b)
                elif activation_function == 1:
                    return tf.nn.relu(Wx_plus_b)
                elif activation_function == 2:
                    return tf.nn.tanh(Wx_plus_b)
                elif activation_function == 3:
                    return tf.nn.softmax(Wx_plus_b)

    def fit(self, net, lr=0.01, aFuncs=None, optimizer=0, epoch=1000, lossFunc=0, keep_prob=1):
        #加载数据
        # boston = load_boston()
        global y_test
        if self.data_path.text().endswith('iris.csv'):

            csv_file = './data/iris.csv'
            self.x_data,self.y_data = self.iris_data(csv_file)
            X_train,X_test,y_train,y_test = train_test_split(self.x_data,self.y_data,test_size=0.3,random_state=0)
        else:
            test_rate = float(self.dsb_testRate.text())
            X_train,X_test,y_train,y_test = train_test_split(self.x_data,self.y_data,test_size=test_rate,random_state=0)
            X_train = scale(X_train)
            X_test = scale(X_test)
            y_train = scale(y_train)
            y_test = scale(y_test)
            # scaler = StandardScaler() #数据标准化
            # X_train = scaler.fit_transform(X_train)
            # X_test = scaler.fit_transform(X_test)
            # # print('pre_scale y_train:',y_test)
            # y_train = scaler.fit_transform(y_train.reshape((-1,int(self.outputLayer.text()))))
            # y_test = scaler.fit_transform(y_test.reshape((-1,int(self.outputLayer.text()))))
            # y_test_origin = scaler.inverse_transform(y_test) #还原原始数据
            # print('scale y_train:',y_test)




        # model-神经网络结构
        global xs
        xs = tf.placeholder(shape=[None, net[0]], dtype=tf.float32, name="inputs")
        ys = tf.placeholder(shape=[None, net[-1]], dtype=tf.float32, name="y_true")
        keep_prob_s = tf.placeholder(dtype=tf.float32)

        with tf.name_scope('hide_layer') as scope:
            hide_name = []
            hide_aFuncs = []
            h_0 = self.add_layer(xs,net[0],net[1],activation_function=aFuncs[0])
            hide_name.append(h_0)
            if len(self.hideLayers) > 0:
                for i in range(1,len(self.hideLayers)+1):
                    hide_name.append('h_%s'%(i+1))
                    hide_aFuncs.append('aFuncs_%s'%(i+1))
                    hide_name[i] = self.add_layer(hide_name[i-1],net[i],net[i+1],activation_function=aFuncs[i])

        with tf.name_scope('pred') as scope:
            global pred
            pred = self.add_layer(hide_name[-1], net[-2], net[-1],activation_function=aFuncs[-1])


        # 这里多于的操作，是为了保存pred的操作，做恢复用。我只知道这个笨方法。
        # pred = tf.add(pred,0,name='pred')

        with tf.name_scope("loss") as scope:
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred), reduction_indices=[1]))  # L2/mse均方误差
            tf.summary.scalar("loss", tensor=loss)


        with tf.name_scope('train_step') as scope:
            if optimizer == 0:
                train_step = tf.train.AdamOptimizer(lr).minimize(loss)
            elif optimizer == 1:
                train_step = tf.train.AdagradOptimizer(lr).minimize(loss)
            # elif optimizer == 2:
            #     train_step = tf.train.AdagradDAOptimizer(lr,epoch).minimize(loss)
            elif optimizer == 3:
                train_step = tf.train.FtrlOptimizer(lr).minimize(loss)
            elif optimizer == 4:
                train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
            elif optimizer == 5:
                train_step = tf.train.MomentumOptimizer(lr,0.9).minimize(loss)
            elif optimizer == 6:
                train_step = tf.train.ProximalAdagradOptimizer(lr).minimize(loss)
            elif optimizer == 7:
                train_step = tf.train.ProximalGradientDescentOptimizer(lr).minimize(loss)
            elif optimizer == 8:
                train_step = tf.train.RMSPropOptimizer(lr).minimize(loss)

        access = tf.equal(tf.argmax(pred,1),tf.argmax(ys,1))
        accuracy = tf.reduce_mean(tf.cast(access,'float'))



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
                # # shuffle
                X_train, y_train = shuffle(X_train, y_train)
                feed_dict_train = {xs: X_train, ys: y_train, keep_prob_s: keep_prob}
                feed_dict_test = {xs: X_test, keep_prob_s: keep_prob}
                loss_value, _ = sess.run([loss, train_step], feed_dict=feed_dict_train)


                if i % int(self.HS_LossStep.value()) == 0:
                    loss_.append(loss_value)
                    self.canvas.setData(loss_[1:])
                    self.plt1.plotItem.setTitle('Loss:%s' % loss_value)
                    global prediction_value
                    prediction_value = sess.run(pred, feed_dict=feed_dict_test)
                    if self.cbb_lossFunc.currentIndex()==0:
                        evaluate = loss_value
                    elif self.cbb_lossFunc.currentIndex()==1:
                        evaluate = loss_value**0.5
                    elif self.cbb_lossFunc.currentIndex()==2:
                        evaluate = np.sum(np.absolute(prediction_value-y_test))/len(y_test)
                    elif self.cbb_lossFunc.currentIndex()==3:
                        evaluate = r2_score(y_test, prediction_value)
                    else:
                        evaluate = None
                    status = '<font color=\'#FFFFFF\'> Epoch: [%.4d/%d], Loss:%g, evaluate:%.3f </font>' \
                             % (i, epoch, loss_value, evaluate)
                    self.textEdit_trainResult.append(status)
                    # prediction_value = scaler.inverse_transform(prediction_value) #转换成与归一化前相对应的数据
                    self.pg_plot()

                    QApplication.processEvents()  # 实时处理
                    time.sleep(0.01)

                    # rs = sess.run(merged,feed_dict=feed_dict_train)
                    # writer.add_summary(summary=rs, global_step=i)  # 写tensorbord
                    # saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=i)  # 保存模型

            # if self.cbb_aFunc_output.currentIndex() == 3:
            #     train_acc = sess.run(accuracy, {xs: X_train, ys: y_train})
            #     test_acc = sess.run(accuracy, {xs: X_test, ys: y_test})
            #     self.textEdit_trainResult.append('训练精度:%s' % train_acc)
            #     self.textEdit_trainResult.append('预测精度:%s' % test_acc)
            #     print('训练集准确率：', train_acc)
            #     print('测试集准确率：', test_acc)

            # 提示音
            # winsound.Beep(300, 1000)
            self.textEdit_trainResult.append('<font color=\'#ffaa00\'>训练完毕！</font>')
            # 保存模型
            if self.checkBox_saveModel.isEnabled():
                # saver.save(sess=sess, save_path="./model/model.ckpt", global_step=epoch)  # 保存模型
                saver.save(sess,'./model/model.ckpt',global_step=epoch)
                self.textEdit_trainResult.append('<font color=\'#ffaa00\'>模型已保存！</font>')

            self.textEdit_trainResult.append('<font color=\'#e6db74\'>★</font>' * 24)
            # 验证
            # global prediction_value
            # prediction_value = sess.run(pred, feed_dict=feed_dict_test)
            # self.pg_plot()



    def pg_plot(self):
        global y_test,prediction_value
        # self.plt2.plotItem.setTitle()
        self.plt2.addLegend(size=(80, 50), offset=(-30, 30))
        # self.plt2.plot(prediction_value[:,0],pen='r',symbol='star', symbolBrush=(237,177,32),name='pred',clear=True)
        # self.plt2.plot(y_test[:,0],pen='g',symbol='h',symbolBrush=(217,83,25),name='real')

        self.plt2.plot(prediction_value[:, 0], pen='g',  name='pred', clear=True)
        self.plt2.plot(y_test[:, 0],pen=None,symbol='star',symbolBrush=(0,0,255),name='real')


    ####################################################
    ################## 专 家 推 理 ######################
    ####################################################
    def similarity(self,component_database, search_values, w):
        '''
        component_database:零件特征
        search_values：待匹配特征
        w：特征权重值
        '''
        l = len(component_database)
        # 加权相似度
        similar_value = 0
        for i in range(l):
            S = 1 - abs(component_database[i] - search_values[i]) / component_database[i]
            if S >= 0:
                similar_value += w[i] * S
        return similar_value

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

    def on_case_search(self):
        # m = [720, 1.3, 924, 14, 35, 170, 103, 19.2]
        # w = [0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.05, 0.05]

        self.textEdit_inference.append('<font color=\'#0000FF\'>正在检索案例···</font>')
        QApplication.processEvents()  # 实时处理
        time.sleep(1)  # 延迟

        file_dir = 'data/structure.csv'
        df = pd.read_csv(file_dir)
        df = np.array(df.values)
        # 特征权值
        w = [0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.05, 0.05]
        m = []
        try:
            m.append(float(self.length.text()))
            m.append(float(self.ASR.text()))
            m.append(float(self.radius.text()))
            m.append(float(self.line1.text()))
            m.append(float(self.line3.text()))
            m.append(float(self.angle2.text()))
            m.append(float(self.angle4.text()))
            m.append(float(self.springback.text()))
        except Exception as e:
            self.textEdit_inference.append('<font color=\'#ff0000\'>Error : Please enter the correct parameters.</font>')
            self.textEdit_inference.append(' ')
            QMessageBox.critical(self,'错误','请输入正确参数！！！')


        try:
            if len(m) == df.shape[1]:
                max_num = 0
                max_similar_value = 0
                for i in range(df.shape[0]):
                    similar_value = self.similarity(df[i, :], m, w)
                    # print('相似零件编号：%d,相似度：%.2f%%' % (i, similar_value * 100))
                    if similar_value > max_similar_value:
                        max_num = i
                        max_similar_value = similar_value
                search_result = '最佳匹配零件：No.%d, 相似度：%.2f%%' % (max_num+1, max_similar_value * 100)
                self.textEdit_inference.append(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                self.textEdit_inference.append('<font color=\'#238e23\'>%s\n</font>'%search_result)
                self.textEdit_inference.append(' ')
        except Exception as e:
            self.textEdit_inference.append('<font color=\'#ff0000\'>Error : %s</font>' % e)
            QMessageBox.critical(self,'错误','Error : %s'%e)

    def on_NN_Inference(self):
        # 预测
        # x_pred = np.array([4.61841693e-04, 0.00000000e+00, 4.20454545e-01, 0.00000000e+00,
        #                     3.86831276e-01, 4.73079134e-01, 8.02265705e-01, 1.25071611e-01,
        #                     0.00000000e+00, 1.64122137e-01, 8.93617021e-01, 1.00000000e+00,
        #                     1.69701987e-01]).reshape(-1,13)
        # y_real = 0.15333333
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

                xs = graph.get_tensor_by_name('inputs:0')
                ys = graph.get_tensor_by_name('y_true:0')
                pred = graph.get_tensor_by_name('pred/Wx_plus_b/Add:0')
                # weights1 = graph.get_tensor_by_name('hide_layer/Weights/weights:0')
                # weights2 = graph.get_tensor_by_name('pred/Weights/weights:0')
                # biases = graph.get_tensor_by_name('hide_layer/biases/biases:0')
                # loss = graph.get_tensor_by_name('loss:0')
                # optimizer = graph.get_tensor_by_name('train_step:0')

                pred_value = sess.run(pred, feed_dict={xs:x_test})
                # print('pre_value:',pred_value)
                # print(graph.get_operations()) #查看参数名


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

                weights1 = graph.get_tensor_by_name('hide_layer/Weights/weights:0')
                weights2 = graph.get_tensor_by_name('pred/Weights/weights:0')

                w1,w2 = sess.run((weights1,weights2))

                #权重系数计算
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
                # print(label)
                # label = ['Length', 'ASR', 'radius', 'line1', 'line2', 'line3', 'line5', 'angle1', 'angle2', 'angle3',
                #           'angle4', 'corner3', 'corner4']
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