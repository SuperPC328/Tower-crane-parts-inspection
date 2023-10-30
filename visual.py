from UI.UI_new import Ui_MainWindow
from UI.UI_waiting import Ui_QWaiting
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import predict_class
import numpy as np
from utils.utils import get_classes
import traceback
import sift

#http://192.168.0.194:8085/?action=stream
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
classes_path = 'model_data/TD_classes.txt'
data_list, _ = get_classes(classes_path)


# 预测线程, 调用yolo模型进行推理
class PredictThread(QThread):
    # 自定义信号对象。参数str就代表这个信号可以传一个字符串
    trigger = pyqtSignal(int)

    def __init__(self, mainwin):
        # 初始化函数
        # super().__init__()
        super(PredictThread, self).__init__()
        self.mainwin = mainwin

    def run(self):
        # 重写线程执行的run函数
        try:
            self.mainwin.model_class.run(self.mainwin.mode, self.mainwin.source, self.mainwin.save_path,
                                         self.mainwin.classes)
            # 触发自定义信号
        except Exception as e:
            # print('ERROR: %s' %(e))
            print(traceback.print_exc())
        self.trigger.emit(1)


# 等待窗体
class WaitWin(Ui_QWaiting, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowOpacity(0.8)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setWindowModality(Qt.WindowModal)
        self.label.setStyleSheet("background-color: transparent")

        # set qmovie as label
        self.movie = QMovie("image/5-121204193957.gif")
        self.label.setMovie(self.movie)
        self.movie.start()
        self.label.setScaledContents(True)
        self.show()

    # def __del__(self):
    #     self.m_Move.stop()


# 主窗体
class MainWin(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # self.setWindowTitle("标题")

        desktop = QApplication.desktop()
        # 界面大小比例
        min_ratio = 0.5
        max_ratio = 0.7
        self.setGeometry(int((1 - max_ratio) * desktop.width() / 2), int((1 - max_ratio) * desktop.height() / 2),
                         int(max_ratio * desktop.width()), int(max_ratio * desktop.height()))
        # connect 方法可以设置对应控件(按钮等)调用哪个函数
        self.PB_import.clicked.connect(self.importMedia)  # 打开按钮
        self.PB_predict.clicked.connect(self.run)  # 预测按钮
        self.PB_predict.setEnabled(True)
        self.PB_stop.clicked.connect(self.stopPredict)  # 停止按钮
        self.PB_resize.clicked.connect(self.resize_label)  # 适应窗口按钮
        self.stop = 0  # 视频/摄像头时起作用, 当stop == 1时, 视频/摄像头停止播放
        self.pushButton_2.clicked.connect(self.on_tbn_max_clicked) #最大化与还原
        # self.video = False

        self.save_path = ''
        self.source = None
        self.model_class = predict_class.UIDetecter(self)  # yolov5模型

        self.predictThread = PredictThread(self)  # 初始化预测线程
        self.predictThread.trigger.connect(self.isdone)

        self.insert()
        self.classes = None

        self.inputBox = MultiInPutDialog()
        self.rtsp1 = ''
        self.rtsp2 = ''


    def insert(self):
        self.listWidget.clear()
        for i in data_list:
            box = QCheckBox(i)  # 实例化一个QCheckBox，把文字传进去
            box.setChecked(True)
            item = QListWidgetItem()  # 实例化一个QListWidgetItem，不能直接加入QCheckBox
            self.listWidget.addItem(item)  # 把QListWidgetItem加入QListWidget
            self.listWidget.setItemWidget(item, box)  # 再把QCheckBox加入QListWidgetItem

    def getClasses(self):
        count = self.listWidget.count()  # 得到QListWidget的总个数
        CB_list = [self.listWidget.itemWidget(self.listWidget.item(i))
                   for i in range(count)]  # 得到QListWidget里面所有QListWidgetItem中的QCheckBox
        # print(cb_list)
        self.classes = []  # 存放被选择的数据
        for i, cb in enumerate(CB_list):  # type:QCheckBox
            if cb.isChecked():
                self.classes.append(cb.text())

    def resizeImg(self, image):
        '''
        调整图片到合适大小
        '''
        width = image.width()  ##获取图片宽度
        height = image.height()  ##获取图片高度
        print(2)
        if width / self.labelsize[1] >= height / self.labelsize[0]:  ##比较图片宽度与label宽度之比和图片高度与label高度之比
            ratio = width / self.labelsize[1]
        else:
            ratio = height / self.labelsize[0]
        new_width = width / ratio  ##定义新图片的宽和高
        new_height = height / ratio
        new_img = image.scaled(new_width, new_height)  ##调整图片尺寸
        return new_img

    def padding(self, image):
        '''
        图片周围补0以适应label大小
        '''
        width = image.shape[1]
        height = image.shape[0]
        target_ratio = self.labelsize[0] / self.labelsize[1]  # h/w
        now_ratio = height / width
        if target_ratio > now_ratio:
            # padding h
            new_h = int(target_ratio * width)
            padding_image = np.ones([int((new_h - height) / 2), width, 3], np.uint8) * 255
            new_img = cv2.vconcat([padding_image, image, padding_image])
        else:
            # padding w
            new_w = int(height / target_ratio)
            padding_image = np.ones([height, int((new_w - width) / 2), 3], np.uint8) * 255
            new_img = cv2.hconcat([padding_image, image, padding_image])
        return new_img

    def resize_label(self):
        '''
        更新label中的图片大小
        '''
        print(1)
        self.labelsize = [self.label_in.height(), self.label_in.width()]
        img_in = self.label_in.pixmap()
        img_out = self.label_out.pixmap()
        try:
            img_in = self.resizeImg(img_in)
        except:
            return
        else:
            self.label_in.setPixmap(img_in)

        try:
            img_out = self.resizeImg(img_out)
        except:
            return
        else:
            self.label_out.setPixmap(img_out)

    def importMedia(self):
        '''
        打开检测源
        '''

        self.labelsize = [self.label_in.height(), self.label_in.width()]

        # 源为摄像头
        if self.RB_camera.isChecked():
            # self.stackedWidget.setCurrentIndex(0)  #切换页面
            self.source = [0]
            self.mode = 'video'
            self.run()

        # 源为sift
        elif self.RB_sift.isChecked():


            # self.stackedWidget.setCurrentIndex(0)  # 切换页面
            # 第一张图
            fname1, _ = QFileDialog.getOpenFileName(self, "打开文件", ".")
            # print(fname)
            if fname1.split('.')[-1].lower() in (IMG_FORMATS):
                # self.importImg(fname1)
                # self.source = fname1
                self.mode = 'image'
            else:
                print('<font color=red>不支持该类型文件...</font>')

            # 第二张图
            fname2, _ = QFileDialog.getOpenFileName(self, "打开文件", ".")
            if fname2.split('.')[-1].lower() in (IMG_FORMATS):
                # self.importImg(fname2)
                # self.source = fname2
                self.mode = 'image'
            else:
                print('<font color=red>不支持该类型文件...</font>')

            self.pQwait = WaitWin()
            self.pQwait.show()

            # SIFT
            sift.start_sift(fname1, fname2)

            self.pQwait.close()

            self.importImg('hhh.jpg')
            self.source = 'hhh.jpg'
            self.mode = 'image'

        # 源为图片/视频
        elif self.RB_img.isChecked():
            # self.stackedWidget.setCurrentIndex(0)  # 切换页面
            fname, _ = QFileDialog.getOpenFileName(self, "打开文件", ".")
            # print(fname)
            if fname.split('.')[-1].lower() in (IMG_FORMATS):
                self.importImg(fname)
                self.source = fname
                self.mode = 'image'
            elif fname.split('.')[-1].lower() in VID_FORMATS:
                self.importImg(fname)
                self.source = [fname]
                self.mode = 'video'
            else:
                print('<font color=red>不支持该类型文件...</font>')
        elif self.RB_rtsp.isChecked():
            # self.stackedWidget.setCurrentIndex(1)  # 切换页面
            if self.getrtsp():
                print('rtsp1: ' + self.rtsp1)
                print('rtsp2: ' + self.rtsp2)
                self.mode = 'video'
                self.source = []
                if self.rtsp1 != '' and self.rtsp1 != '\n' and self.rtsp1 != None:
                    self.source.append(self.rtsp1)
                if self.rtsp2 != '' and self.rtsp2 != '\n' and self.rtsp2 != None:
                    self.source.append(self.rtsp2)
                self.run()

        else:
            print('<font color=red>请选择检测源类型...</font>')

    def importImg(self, file_name):
        '''
        label_in 中显示图片/视频第一帧
        '''
        if file_name.split('.')[-1].lower() in VID_FORMATS:
            cap = cv2.VideoCapture(file_name)
            if cap.isOpened():
                # self.video = True
                ret, img_in = cap.read()
                if ret:
                    img_in = self.padding(img_in)
                    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
                    # padding
                    img_in = QImage(img_in, img_in.shape[1], img_in.shape[0], QImage.Format_RGB888)
                    img_in = QPixmap(img_in)
            cap.release()
        elif file_name.split('.')[-1].lower() in IMG_FORMATS:
            # self.video = False
            img_in = cv2.imread(file_name)
            img_in = self.padding(img_in)
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGBA)
            img_in = QImage(img_in, img_in.shape[1], img_in.shape[0], QImage.Format_RGBA8888)
            img_in = QPixmap(img_in)
        if img_in.isNull():
            print('<font color=red>打开失败...</font>')
            return
        # img_in = img_in.scaledToWidth(self.labelsize[1])
        img_in = self.resizeImg(img_in)
        self.label_in.setPixmap(img_in)

    def stopPredict(self):
        '''
        stop == 1 播放停止
        '''
        self.stop = 1

    def run(self):
        '''
        开始预测
        '''
        if self.source == None:
            print('<font color=red>请选择检测源...</font>')
            return
        else:
            try:
                self.getClasses()
                self.predictThread.start()
                self.PB_predict.setEnabled(False)
                self.PB_stop.setEnabled(True)
            except:
                print('ERROR!')

    def isdone(self, done):
        '''
        结束一次预测
        '''
        if done == 1:
            self.PB_predict.setEnabled(True)
            # self.PB_import.setEnabled(True)
            self.PB_stop.setEnabled(False)
            self.stop = 0
            self.predictThread.quit()

    def getrtsp(self):
        self.inputBox.show()
        if self.inputBox.exec_():
            self.rtsp1 = self.inputBox.rtsp1_line.text()
            self.rtsp2 = self.inputBox.rtsp2_line.text()
            return True
        else:
            return False

    # 拖动
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, mouse_event):
        if Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    #判断窗体最大化
    def on_tbn_max_clicked(self):
        if self.isMaximized():
            self.showNormal()
            self.pushButton_2.setIcon(QIcon("icon/方框.png"))

        else:
            self.showMaximized()
            self.pushButton_2.setIcon(QIcon("icon/最大化.png"))


class MultiInPutDialog(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.resize(100, 80)
        self.setWindowTitle('Input')

        grid = QGridLayout()
        grid.addWidget(QLabel('RTSP1:', parent=self), 0, 0, 1, 1)
        self.rtsp1_line = QLineEdit(parent=self)
        grid.addWidget(self.rtsp1_line, 0, 1, 1, 1)

        grid.addWidget(QLabel('RTSP2:', parent=self), 1, 0, 1, 1)
        self.rtsp2_line = QLineEdit(parent=self)
        grid.addWidget(self.rtsp2_line, 1, 1, 1, 1)

        buttonBox = QDialogButtonBox(parent=self)
        buttonBox.setOrientation(Qt.Horizontal)  # 设置为水平方向
        buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)  # 确定和取消两个按钮

        buttonBox.accepted.connect(self.accept)  # 确定
        buttonBox.rejected.connect(self.reject)  # 取消

        layout = QVBoxLayout()
        layout.addLayout(grid)

        spacerItem = QSpacerItem(20, 48, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacerItem)
        layout.addWidget(buttonBox)
        self.setLayout(layout)

    # -------------------Close Event Method----------------------
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Close Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWin()
    main.show()

    sys.exit(app.exec_())
