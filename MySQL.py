import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from login import Ui_login

import pymysql


class login(QWidget,Ui_login):
    def __init__(self):
        super(login, self).__init__()
        self.setupUi(self)
        # self.setWindowFlag(Qt.FramelessWindowHint)  # 隐藏框¡
        self.initUI()
        self.initConnect()

    def initUI(self):
        pass

    def initConnect(self):
        self.btn_login.clicked.connect(self.register)

    def register(self):
        print('连接中...')
        user = self.user.text()
        password = self.password.text()
        database = self.cbb_database.currentText()

        try:
            connect = pymysql.connect(user=user,password=password,database=database,charset='utf8')
        except:
            pass
        cursor = connect.cursor()
        query = ('select id,name  from stu')
        cursor.execute(query)
        for (id,name) in cursor:
            print(id,name)

        cursor.close()
        connect.close()

# connect = pymysql.connect(user='root',password='123456',database='test',charset='utf8')
# cursor = connect.cursor()
# query = ('select id,name from stu')
# cursor.execute(query)
# for (id,name) in cursor:
#     print(id,name)
#
# cursor.close()
# connect.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = login()
    w.show()
    sys.exit(app.exec_())