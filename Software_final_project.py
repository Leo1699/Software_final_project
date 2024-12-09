import cv2
import numpy as np
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

    def get_sqinfo(self, image):
        """提取方块锚点和间距信息"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img1 = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(img1, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
        horizontal_lines = []
        vertical_lines = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if rho < 0:
                    continue
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                if 0 <= int(theta * 180 / np.pi) <= 2 or 178 <= int(theta * 180 / np.pi) <= 175:
                    horizontal_lines.append(int(x0))
                elif 88 <= int(theta * 180 / np.pi) <= 92:
                    vertical_lines.append(int(y0))
        horizontal_lines.sort()
        vertical_lines.sort()
        gaps = []
        for i in range(len(horizontal_lines) - 1):
            gaps.append(horizontal_lines[i + 1] - horizontal_lines[i])
        cnt = Counter(gaps)
        hwidth = max(cnt, key=cnt.get)
        hgap = min(cnt, key=cnt.get)
        gaps = []
        for i in range(len(vertical_lines) - 1):
            gaps.append(vertical_lines[i + 1] - vertical_lines[i])
        cnt = Counter(gaps)
        vwidth = max(cnt, key=cnt.get)
        vgap = min(cnt, key=cnt.get)
        anchor_x = horizontal_lines[0]
        anchor_y = vertical_lines[0]
        self.sqinfo = {
            'anchor_x': anchor_x,
            'anchor_y': anchor_y,
            'hwidth': hwidth,
            'vwidth': vwidth,
            'hgap': hgap,
            'vgap': vgap,
            'h': hgap + hwidth,
            'v': vgap + vwidth
        }
        return self.sqinfo

    def crop_region(self, square):
        """裁剪区域"""
        (x1, y1, x2, y2) = square
        cropped_region = self.image[y1:y2, x1:x2]
        return cropped_region

    def get_matrix(self, image):
        """生成数字矩阵"""
        self.image = image
        sqinfo = self.get_sqinfo(image)
        squares = []
        for i in range(16):
            for j in range(10):
                squares.append((sqinfo['anchor_x'] + j * sqinfo['h'],
                                sqinfo['anchor_y'] + i * sqinfo['v'],
                                sqinfo['anchor_x'] + sqinfo['hwidth'] + j * sqinfo['h'],
                                sqinfo['anchor_y'] + sqinfo['vwidth'] + i * sqinfo['v']))
        self.crop_images = list(map(self.crop_region, squares))
        recognized_digits = list(map(recognize_digit, self.crop_images))
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

    def cal_two_x(self, action=False):
        """两数和为10，行优先（上下左右搜索）"""
        for begin_x in range(0, 16):
            for begin_y in range(0, 10):
                if self.cal_matrix[begin_x, begin_y] == 0:
                    continue
                # 搜索右侧
                for x in range(begin_x + 1, 16):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({x}, {begin_y})")
                        break
                    else:
                        break
                # 搜索左侧
                for x in range(begin_x - 1, -1, -1):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({x}, {begin_y})")
                        break
                    else:
                        break
                # 搜索下方
                for y in range(begin_y + 1, 10):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({begin_x}, {y})")
                        break
                    else:
                        break
                # 搜索上方
                for y in range(begin_y - 1, -1, -1):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({begin_x}, {y})")
                        break
                    else:
                        break

    def cal_two_y(self, action=False):
        """两数和为10，列优先（上下左右搜索）"""
        for begin_y in range(0, 10):
            for begin_x in range(0, 16):
                if self.cal_matrix[begin_x, begin_y] == 0:
                    continue
                # 搜索右侧
                for x in range(begin_x + 1, 16):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({x}, {begin_y})")
                        break
                    else:
                        break
                # 搜索左侧
                for x in range(begin_x - 1, -1, -1):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({x}, {begin_y})")
                        break
                    else:
                        break
                # 搜索下方
                for y in range(begin_y + 1, 10):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({begin_x}, {y})")
                        break
                    else:
                        break
                # 搜索上方
                for y in range(begin_y - 1, -1, -1):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({begin_x}, {y})")
                        break
                    else:
                        break

    def execute_strategy(self, strategy_name):
        """根据策略执行操作"""
        self.actions.clear()
        self.cal_matrix = self.matrix.copy()
        if strategy_name == "两数和行优先":
            self.cal_two_x(action=True)
        elif strategy_name == "两数和列优先":
            self.cal_two_y(action=True)
        return self.actions


if __name__ == "__main__":
    # 读取截图
    screenshot = cv2.imread("screenshot.png")

    # 识别数字矩阵
    recognizer = Recognizer()
    matrix = recognizer.get_matrix(screenshot)

    # 初始化 Eliminater
    eliminater = Eliminater(matrix)

    # 策略计算
    strategies = ["两数和行优先", "两数和列优先"]
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
