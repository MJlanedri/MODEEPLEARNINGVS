import traceback
from pprint import PrettyPrinter

from PyQt6.QtCore import QFile, QIODevice
from PyQt6.QtWidgets import QApplication

pp = PrettyPrinter(indent=4).pprint
def dumpException(e):
    print("EXCEPTION:", e)
    traceback.print_tb(e.__traceback__)


def loadStylesheet(filename):
    print('STYLE loading: ', filename)
    file = QFile(filename)
    file.open(QIODevice.OpenModeFlag.ReadOnly | QIODevice.OpenModeFlag.Text)
    styleSheet = file.readAll()
    QApplication.instance().setStyleSheet(str(styleSheet, 'utf-8'))

def loadStylesheets(*args):
    res = ''
    for arg in args:
        file = QFile(arg)
        file.open(QIODevice.OpenModeFlag.ReadOnly | QIODevice.OpenModeFlag.Text)
        styleSheet = file.readAll()
        res += "\n" + str(styleSheet, 'utf-8')
    QApplication.instance().setStyleSheet(res)
