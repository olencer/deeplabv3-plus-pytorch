from PyQt5.QtCore import QSize, QRect
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QPushButton, QLabel, QVBoxLayout

import os
import sys
import cv2

import utils.data as data
import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('航空发动机部件分割预测结果展示')
        self.setGeometry(100, 100, 1400, 900)
        
        self.initUI()
    
    def initUI(self):
        font = QFont()
        font.setFamily(u"\u971e\u9e5c\u6587\u6977 Light")
        font.setPointSize(20)
    
        self.open_button = self.createButton('选择文件', self.open_file_dialog)

        self.nameLbl1 = QLabel(self)
        self.nameLbl1.setObjectName(u"nameLbl1")
        self.nameLbl1.setGeometry(QRect(290, 40, 101, 61))
        self.nameLbl1.setFont(font)
        self.nameLbl1.setText('RGB图')
        self.nameLbl1.hide()

        self.rgbLbl = QLabel(self)
        self.rgbLbl.setObjectName(u"rgbLbl")
        self.rgbLbl.setGeometry(QRect(100, 110, 480, 300))
        self.rgbLbl.setMinimumSize(QSize(321, 0))
        self.rgbLbl.hide()

        self.nameLbl2 = QLabel(self)
        self.nameLbl2.setObjectName(u"nameLbl2")
        self.nameLbl2.setGeometry(QRect(290, 420, 101, 61))
        self.nameLbl2.setFont(font)
        self.nameLbl2.setText('深度图')
        self.nameLbl2.hide()
        
        self.depthLbl = QLabel(self)
        self.depthLbl.setObjectName(u"depthLbl")
        self.depthLbl.setGeometry(QRect(100, 490, 480, 300))
        self.depthLbl.setMinimumSize(QSize(321, 0))
        self.depthLbl.hide()

        self.nameLbl3 = QLabel(self)
        self.nameLbl3.setObjectName(u"nameLbl3")
        self.nameLbl3.setGeometry(QRect(990, 230, 101, 61))
        self.nameLbl3.setFont(font)
        self.nameLbl3.setText('预测')
        self.nameLbl3.hide()

        self.predictLbl = QLabel(self)
        self.predictLbl.setObjectName(u"predictLbl")
        self.predictLbl.setGeometry(QRect(800, 310, 480, 300))
        self.predictLbl.setMinimumSize(QSize(321, 0))
        self.predictLbl.hide()

    def createButton(self, label, callback):
        btn = QPushButton(label, self)
        btn.clicked.connect(callback)
        btn.resize(btn.sizeHint())
        return btn
    
    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "",
                                              "Zivid Point Cloud Format (*.zdf)", options=options)
        input = data.Data(fileName)
        cv2.imwrite(os.path.join('visualize', 'rgb.jpg'), input.get_rgb())
        cv2.imwrite(os.path.join('visualize', 'depth.png'), input.get_depth())

        rgbQImg = QImage(os.path.join('visualize', 'rgb.jpg')).scaled(480,300)
        depthQImg = QImage(os.path.join('visualize', 'depth.png')).scaled(480,300)

        self.rgbLbl.setPixmap(QPixmap.fromImage(rgbQImg))
        self.nameLbl1.show()
        self.rgbLbl.show()
        
        self.depthLbl.setPixmap(QPixmap.fromImage(depthQImg))
        self.nameLbl2.show()
        self.depthLbl.show()

        self.predict()

        predictQImg = QImage(os.path.join('visualize', 'predict.jpg')).scaled(480,300)
        self.predictLbl.setPixmap(QPixmap.fromImage(predictQImg))
        self.nameLbl3.show()
        self.predictLbl.show()

    def predict(self):

        deeplab = DeeplabV3()
        count           = False
        name_classes    = ["background","buckle","tuber"]

        rgb = Image.open(os.path.join('visualize', 'rgb.jpg'))
        depth = Image.open(os.path.join('visualize', 'depth.png'))
        r_image = deeplab.detect_image(rgb, count=count, name_classes=name_classes)
        r_image.save(os.path.join('visualize', 'predict.jpg'))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())