# author:"ZPC"
# date:2022/8/4 11:24

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# 检测图像关键点
def sift_detect(img):
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detect(img, None)


# 展示图像
def imshow(img, title=None, dpi=200):
    plt.figure(dpi=dpi)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.imshow(img[..., ::-1])
    plt.show()


# 无裁剪的旋转
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return M, cv2.warpAffine(image, M, (nW, nH))


# 计算经线性变换后的点
def apply_matrix(M, point):
    point = np.array([point[0], point[1], 1]).T
    M = np.vstack([M, [0, 0, 1]])
    res = M.dot(point)
    return (int(res[0]), int(res[1]))


# 统计匹配的特征点
def count_matched_kp(kp_set, target_kp_set):
    d = [(i % 3 - 1, j % 3 - 1) for i in range(3) for j in range(3)]
    # d = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    count = 0
    for i in target_kp_set:
        for j in d:
            if (i[0] + j[0], i[1] + j[1]) in kp_set:
                count += 1
                break
    return count


class Stitcher:
    # 拼接函数
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):

        # 获取输入图片
        (imageB, imageA) = images
        # 检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # 防止卡死界面
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)


        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None
        # 否则，提取匹配结果
        # H是3x3视角变换矩阵
        (matches, H, status) = M
        # 将图片A进行视角变换，result是变换后图片
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        # 图像融合
        for r in range(result.shape[0]):
            left = 0
            for c in range(result.shape[1] // 2):
                if result[r, c].any():  # overlap
                    if left == 0:
                        left = c
                    alpha = (c - left) / (result.shape[1] // 2 - left)
                    result[r, c] = imageB[r, c] * (1 - alpha) + result[r, c] * alpha

                else:
                    result[r, c] = imageB[r, c]
                # 防止卡死界面
                QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

        # 防止卡死界面
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        # 将图片B传入result图片最左端
        #         result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis)

        # 返回匹配结果
        return result

    def detectAndDescribe(self, image):
        # 将彩色图片转换成灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv2.xfeatures2d.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(gray, None)

        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.DescriptorMatcher_create("BruteForce")

        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        # match匹配的返回结果是DMatch类型。
        #
        # DMatch数据结构包含三个非常重要的数据分别是queryIdx，trainIdx，distance；
        # queryIdx：某一特征点在本帧图像的索引；
        # trainIdx：trainIdx是该特征点在另一张图像中相匹配的特征点的索引；
        # distance：代表这一对匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # 返回结果
            return (matches, H, status)

        # 如果匹配对小于4时，返回None 放弃
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis


def start_sift(fname1,fname2):
    # 读取拼接图片
    imageA = cv2.imread(fname1)
    imageB = cv2.imread(fname2)

    # 防止卡死界面
    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    # 把图片拼接成全景图
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    #imshow(vis,title='Keypoint Matches',dpi=200)
    #imshow(result, title='Result',dpi=200)

    cv2.imwrite('hhh.jpg',result)
    cv2.imwrite('h.jpg',vis)
#start_sift('t1.jpg','t2.jpg')

