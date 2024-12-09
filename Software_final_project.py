import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import pickle
from collections import Counter


# 加载模板
template = pickle.load(open('template.pkl', 'rb'))


def recognize_digit(image):
    """识别单个方块中的数字"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_ = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    scores = np.zeros(10)
    for number, template_img in template.items():
        score = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
        scores[int(number)] = np.max(score)
    if np.max(scores) < 200000:
        print('识别出错！')
    return np.argmax(scores)


class Recognizer:
    """识别模块，负责提取矩阵"""

    def __init__(self):
        self.sqinfo = {}

    # def get_sqinfo(self, image):
    #     """改进提取方块锚点和间距信息"""
    #     # 转为灰度图并检测边缘
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     img1 = cv2.GaussianBlur(gray, (3, 3), 0)
    #     edges = cv2.Canny(img1, 50, 150)
    #
    #     # 使用霍夫线检测
    #     lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)  # 降低阈值提高检测率
    #     horizontal_lines = []
    #     vertical_lines = []
    #     if lines is not None:
    #         for line in lines:
    #             rho, theta = line[0]
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a * rho
    #             y0 = b * rho
    #             angle = int(theta * 180 / np.pi)
    #             # 宽松的水平线和垂直线检测
    #             if abs(angle - 0) <= 5 or abs(angle - 180) <= 5:  # 水平线
    #                 horizontal_lines.append(int(y0))
    #             elif abs(angle - 90) <= 5:  # 垂直线
    #                 vertical_lines.append(int(x0))
    #
    #     # 确保排序
    #     horizontal_lines = sorted(set(horizontal_lines))
    #     vertical_lines = sorted(set(vertical_lines))
    #
    #     if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
    #         raise ValueError("检测的水平线或垂直线不足，无法提取锚点和间距")
    #
    #     # 计算间距
    #     h_gaps = [horizontal_lines[i + 1] - horizontal_lines[i] for i in range(len(horizontal_lines) - 1)]
    #     v_gaps = [vertical_lines[i + 1] - vertical_lines[i] for i in range(len(vertical_lines) - 1)]
    #     hwidth = max(h_gaps, key=h_gaps.count)  # 最常见的水平间距
    #     vwidth = max(v_gaps, key=v_gaps.count)  # 最常见的垂直间距
    #     hgap = min(h_gaps)  # 最小水平间隙
    #     vgap = min(v_gaps)  # 最小垂直间隙
    #
    #     # 选择锚点（左上角的线交点）
    #     anchor_x = vertical_lines[0]
    #     anchor_y = horizontal_lines[0]
    #
    #     self.sqinfo = {
    #         'anchor_x': anchor_x,
    #         'anchor_y': anchor_y,
    #         'hwidth': hwidth,
    #         'vwidth': vwidth,
    #         'hgap': hgap,
    #         'vgap': vgap,
    #         'h': hwidth + hgap,
    #         'v': vwidth + vgap
    #     }
    #
    #     print(f"提取的方块信息: {self.sqinfo}")
    #     return self.sqinfo

    def get_sqinfo(self, image):
        """改进方块锚点和间距提取方法"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 尝试不同的二值化方法
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # 反转二值化
        # 或者使用自适应二值化
        # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 可视化轮廓，调试用
        debug_image = image.copy()
        cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite("debug_contours.png", debug_image)

        # 提取方块的外接矩形
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        rects = sorted(rects, key=lambda x: (x[1], x[0]))  # 按 (y, x) 排序

        if len(rects) < 20:  # 假设至少需要20个轮廓
            raise ValueError("未检测到足够的方块轮廓")

        # 计算锚点和间距
        hwidth = rects[0][2]
        vwidth = rects[0][3]
        h_gaps = [rects[i + 1][0] - rects[i][0] for i in range(len(rects) - 1)]
        v_gaps = [rects[i + 1][1] - rects[i][1] for i in range(len(rects) - 1)]
        hgap = min([gap for gap in h_gaps if gap > hwidth])
        vgap = min([gap for gap in v_gaps if gap > vwidth])

        # 获取左上角第一个方块
        anchor_x, anchor_y = rects[0][0], rects[0][1]

        self.sqinfo = {
            'anchor_x': 16,  # 手动调整
            'anchor_y': 140,  # 手动调整
            'hwidth': 37,
            'vwidth': 37,
            'hgap': 6,
            'vgap': 4,
            'h': 43,
            'v': 41
        }

        print(f"提取的方块信息: {self.sqinfo}")
        return self.sqinfo


    def crop_region(self, square):
        (x1, y1, x2, y2) = square
        if x1 < 0 or y1 < 0 or x2 > self.image.shape[1] or y2 > self.image.shape[0]:
            print(f"无效裁剪区域: {(x1, y1, x2, y2)}")
            return None
        cropped_region = self.image[y1:y2, x1:x2]
        if cropped_region is not None and cropped_region.size > 0:
            cv2.imwrite(f"crop_{x1}_{y1}_{x2}_{y2}.png", cropped_region)  # 保存裁剪结果
        return cropped_region

    def get_matrix(self, image):
        self.image = image
        sqinfo = self.get_sqinfo(image)
        squares = []
        for i in range(16):
            for j in range(10):
                squares.append((sqinfo['anchor_x'] + j * sqinfo['h'],
                                sqinfo['anchor_y'] + i * sqinfo['v'],
                                sqinfo['anchor_x'] + sqinfo['hwidth'] + j * sqinfo['h'],
                                sqinfo['anchor_y'] + sqinfo['vwidth'] + i * sqinfo['v']))
        print(f"生成的裁剪区域数量: {len(squares)}")  # 添加此行
        self.crop_images = list(map(self.crop_region, squares))
        valid_crops = [crop for crop in self.crop_images if crop is not None]
        print(f"有效裁剪区域数量: {len(valid_crops)}")  # 添加此行
        recognized_digits = list(map(recognize_digit, valid_crops))
        self.digits_matrix = []
        for i in range(16):
            self.digits_matrix.append((recognized_digits[i * 10:i * 10 + 10]))
        return self.digits_matrix


class Eliminater:
    """消除模块，负责根据策略操作矩阵"""

    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.cal_matrix = self.matrix.copy()
        self.actions = []

    def score(self):
        """计算当前剩余非零方块的数量"""
        return 160 - np.sum(self.cal_matrix.astype(bool))

    def cal_all_x(self, End=False, action=False):
        """
        任意和为10的连续矩形，行优先的消除逻辑
        """
        if End:
            return
        End = True
        for x_len in range(1, 16):
            for y_len in range(1, 10):
                for begin_x in range(0, 16 - x_len + 1):
                    for begin_y in range(0, 10 - y_len + 1):
                        _sum = np.sum(self.cal_matrix[begin_x:begin_x + x_len, begin_y:begin_y + y_len])
                        if _sum == 10:
                            self.cal_matrix[begin_x:begin_x + x_len, begin_y:begin_y + y_len] = 0
                            if action:
                                self.actions.append(f"消除 ({begin_x}:{begin_x + x_len}, {begin_y}:{begin_y + y_len})")
                            End = False
        self.cal_all_x(End=End, action=action)

    def cal_all_y(self, End=False, action=False):
        """
        任意和为10的连续矩形，列优先的消除逻辑
        """
        if End:
            return
        End = True
        for y_len in range(1, 10):
            for x_len in range(1, 16):
                for begin_x in range(0, 16 - x_len + 1):
                    for begin_y in range(0, 10 - y_len + 1):
                        _sum = np.sum(self.cal_matrix[begin_x:begin_x + x_len, begin_y:begin_y + y_len])
                        if _sum == 10:
                            self.cal_matrix[begin_x:begin_x + x_len, begin_y:begin_y + y_len] = 0
                            if action:
                                self.actions.append(f"消除 ({begin_x}:{begin_x + x_len}, {begin_y}:{begin_y + y_len})")
                            End = False
        self.cal_all_y(End=End, action=action)

    def cal_two_x(self, End=False, action=False):
        """两数和为10，行优先的消除逻辑"""
        if End:
            return
        End = True
        for begin_x in range(0, 16):
            for begin_y in range(0, 10):
                if self.cal_matrix[begin_x, begin_y] == 0:
                    continue
                for x in range(begin_x + 1, 16):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({x}, {begin_y})")
                        End = False
                        break
        self.cal_two_x(End=End, action=action)

    def cal_two_y(self, End=False, action=False):
        """两数和为10，列优先的消除逻辑"""
        if End:
            return
        End = True
        for begin_y in range(0, 10):
            for begin_x in range(0, 16):
                if self.cal_matrix[begin_x, begin_y] == 0:
                    continue
                for x in range(begin_x + 1, 16):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({x}, {begin_y})")
                        End = False
                        break
        self.cal_two_y(End=End, action=action)

    def execute_strategy(self, strategy_name):
        """根据策略执行操作"""
        self.actions.clear()
        self.cal_matrix = self.matrix.copy()
        if strategy_name == "两数和行优先":
            self.cal_two_x(action=True)
        elif strategy_name == "两数和列优先":
            self.cal_two_y(action=True)
        elif strategy_name == "任意和行优先":
            self.cal_all_x(action=True)
        elif strategy_name == "任意和列优先":
            self.cal_all_y(action=True)
        return self.actions


if __name__ == "__main__":
    screenshot = cv2.imread("screenshot.png")

    # 识别数字矩阵
    recognizer = Recognizer()
    matrix = recognizer.get_matrix(screenshot)

    # 将识别到的数字矩阵保存到 TXT 文件
    with open("matrix_output.txt", "w") as file:
        for row in matrix:
            file.write(" ".join(map(str, row)) + "\n")
    print("识别到的数字矩阵已保存到 matrix_output.txt")

    # 初始化 Eliminater
    eliminater = Eliminater(matrix)

    # 策略计算
    strategies = ["两数和行优先", "两数和列优先", "任意和行优先", "任意和列优先"]
    strategy_scores = {}
    strategy_actions = {}

    for strategy in strategies:
        eliminater.execute_strategy(strategy)
        strategy_scores[strategy] = eliminater.score()
        strategy_actions[strategy] = eliminater.actions

    # 找到最高分策略
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    best_actions = strategy_actions[best_strategy]

    # 保存结果
    with open("result.txt", "w") as file:
        file.write(f"最佳策略: {best_strategy}\n")
        file.write(f"得分: {strategy_scores[best_strategy]}\n")
        file.write("消除步骤:\n")
        file.write("\n".join(best_actions))

    print("结果已保存到 result.txt")
