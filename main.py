import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from mainWindows import *
from welcome import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = welcome()
    w.show()
    sys.exit(app.exec_())
