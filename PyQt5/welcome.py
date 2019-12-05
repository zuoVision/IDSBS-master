# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'welcome.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Welcome(object):
    def setupUi(self, Welcome):
        Welcome.setObjectName("Welcome")
        Welcome.resize(640, 480)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Welcome.sizePolicy().hasHeightForWidth())
        Welcome.setSizePolicy(sizePolicy)
        Welcome.setMaximumSize(QtCore.QSize(640, 480))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/background image/image/1.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Welcome.setWindowIcon(icon)
        Welcome.setStyleSheet("color: rgb(14, 28, 41);\n"
"background-color: rgb(14, 28, 41);")
        self.gridLayout_2 = QtWidgets.QGridLayout(Welcome)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(148, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 2, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(148, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(Welcome)
        self.label.setStyleSheet("color: rgb(0, 170, 255);\n"
"font: 75 28pt \"微软雅黑\";")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 3)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.btn_1 = QtWidgets.QPushButton(Welcome)
        self.btn_1.setMinimumSize(QtCore.QSize(150, 150))
        self.btn_1.setMaximumSize(QtCore.QSize(150, 150))
        font = QtGui.QFont()
        font.setFamily("萍方0")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.btn_1.setFont(font)
        self.btn_1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_1.setStyleSheet("image: url(:/image/image/nn.jpg);\n"
"font:75 14pt \"萍方0\";\n"
"background-image: url(:/image/image/nn.jpg);\n"
"\n"
"\n"
"color: rgb(217, 217, 217);")
        self.btn_1.setObjectName("btn_1")
        self.gridLayout.addWidget(self.btn_1, 0, 0, 1, 1)
        self.btn_2 = QtWidgets.QPushButton(Welcome)
        self.btn_2.setMinimumSize(QtCore.QSize(150, 150))
        self.btn_2.setMaximumSize(QtCore.QSize(150, 150))
        font = QtGui.QFont()
        font.setFamily("萍方0")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.btn_2.setFont(font)
        self.btn_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_2.setStyleSheet("image: url(:/image/image/nn.jpg);\n"
"font:75 14pt \"萍方0\";\n"
"background-image: url(:/image/image/nn.jpg);\n"
"\n"
"\n"
"color: rgb(217, 217, 217);")
        self.btn_2.setObjectName("btn_2")
        self.gridLayout.addWidget(self.btn_2, 0, 1, 1, 1)
        self.btn_3 = QtWidgets.QPushButton(Welcome)
        self.btn_3.setMinimumSize(QtCore.QSize(150, 150))
        self.btn_3.setMaximumSize(QtCore.QSize(150, 150))
        font = QtGui.QFont()
        font.setFamily("萍方0")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.btn_3.setFont(font)
        self.btn_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_3.setStyleSheet("image: url(:/image/image/nn.jpg);\n"
"font:75 14pt \"萍方0\";\n"
"background-image: url(:/image/image/nn.jpg);\n"
"\n"
"\n"
"color: rgb(217, 217, 217);")
        self.btn_3.setObjectName("btn_3")
        self.gridLayout.addWidget(self.btn_3, 1, 0, 1, 1)
        self.btn_4 = QtWidgets.QPushButton(Welcome)
        self.btn_4.setMinimumSize(QtCore.QSize(150, 150))
        self.btn_4.setMaximumSize(QtCore.QSize(150, 150))
        font = QtGui.QFont()
        font.setFamily("萍方0")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.btn_4.setFont(font)
        self.btn_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_4.setStyleSheet("image: url(:/image/image/nn.jpg);\n"
"font:75 14pt \"萍方0\";\n"
"background-image: url(:/image/image/nn.jpg);\n"
"\n"
"\n"
"color: rgb(217, 217, 217);")
        self.btn_4.setObjectName("btn_4")
        self.gridLayout.addWidget(self.btn_4, 1, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 1, 2, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem2, 3, 1, 1, 1)

        self.retranslateUi(Welcome)
        QtCore.QMetaObject.connectSlotsByName(Welcome)
        Welcome.setTabOrder(self.btn_1, self.btn_2)
        Welcome.setTabOrder(self.btn_2, self.btn_3)
        Welcome.setTabOrder(self.btn_3, self.btn_4)

    def retranslateUi(self, Welcome):
        _translate = QtCore.QCoreApplication.translate
        Welcome.setWindowTitle(_translate("Welcome", "汽车冲压件智能CAPP系统"))
        self.label.setText(_translate("Welcome", "典型车身冷冲压结构件智能设计系统"))
        self.btn_1.setText(_translate("Welcome", "数据管理"))
        self.btn_2.setText(_translate("Welcome", "智能学习"))
        self.btn_3.setText(_translate("Welcome", "专家推理"))
        self.btn_4.setText(_translate("Welcome", "知识采集"))

import resources_rc
