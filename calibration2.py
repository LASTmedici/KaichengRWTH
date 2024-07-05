import cv2
import numpy as np
import glob

def calibrate_camera(images_path, chessboard_size=(6, 9), square_size=1.0):
    """
    使用一系列棋盘图像校准相机，获得内参矩阵和畸变系数。

    参数：
    - images_path: 包含棋盘图像的文件路径模式，例如 'path/to/images/*.jpg'。
    - chessboard_size: 棋盘的大小 (行数, 列数)，例如 (6, 9)。
    - square_size: 每个棋盘方块的实际尺寸，单位可以是毫米、厘米等。

    返回：
    - camera_matrix: 相机内参矩阵。
    - dist_coeffs: 畸变系数。
    - rvecs: 旋转向量。
    - tvecs: 平移向量。
    """
    # 准备棋盘格点的世界坐标 (0, 0, 0), (1, 0, 0), (2, 0, 0), ...
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # 用于存储棋盘格的3D点和2D图像点
    objpoints = [] # 3D点
    imgpoints = [] # 2D图像点

    # 读取所有图像文件
    images = glob.glob(images_path)

    for fname in images[::10]:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_size[0], chessboard_size[1]), None)

        # 如果找到足够的角点，则添加到列表中
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # 绘制并显示角点
            cv2.drawChessboardCorners(img, (chessboard_size[0], chessboard_size[1]), corners, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(1)

        # 执行相机标定
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return camera_matrix, dist_coeffs, rvecs, tvecs

# 示例用法
images_path = 'F:/rawdata_20240703/cam/cam1/*.bmp'
camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(images_path)

print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)