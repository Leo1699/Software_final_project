import cv2
import numpy as np
import pickle
from collections import Counter
import json
import sys
import time

# 加载模板
template = pickle.load(open('template.pkl', 'rb'))

def recognize_digit(image):
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
    def __init__(self):
        try:
            self.sqinfo = json.load(open('sqinfo.json','r'))
            print()
            print('从sqinfo.json加载识别模块')
            print(f"左上角方块锚点坐标({self.sqinfo['anchor_x']},{self.sqinfo['anchor_y']})")
            print(f"方块高度{self.sqinfo['hwidth']}, 方块高度间隔{self.sqinfo['hgap']}")
            print(f"方块宽度{self.sqinfo['vwidth']}, 方块宽度间隔{self.sqinfo['vgap']}")
            print()
            return
        except:
            pass

    def get_sqinfo(self, image):
        try:
            return self.sqinfo
        except:
            print()
            print('初始化识别模块，请判断定位是否准确')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img1 = cv2.GaussianBlur(gray,(3,3),0)
        edges = cv2.Canny(img1, 50, 150)
        # 使用霍夫线变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
        horizontal_lines = []
        vertical_lines = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if rho < 0 :
                    continue
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                # 根据角度进行分类，阈值可以根据实际情况调整
                if 0 <= int(theta*180/np.pi) <= 2 or 178 <= int(theta*180/np.pi) <= 182:
                    horizontal_lines.append(int(x0))
                elif 88 <= int(theta*180/np.pi) <= 92:
                    vertical_lines.append(int(y0))
        # 对横线按照从上到下的顺序排序
        horizontal_lines.sort()
        vertical_lines.sort()
        gaps = []
        for i in range(len(horizontal_lines)-1):
            gaps.append(horizontal_lines[i+1] - horizontal_lines[i])
        cnt = Counter(gaps)
        gaps = [cnt.most_common(2)[0][0], cnt.most_common(2)[1][0]]
        hwidth = max(gaps)
        hgap = min(gaps)
        gaps = []
        for i in range(len(vertical_lines)-1):
            gaps.append(vertical_lines[i+1] - vertical_lines[i])
        cnt = Counter(gaps)
        gaps = [cnt.most_common(2)[0][0], cnt.most_common(2)[1][0]]
        vwidth = max(gaps)
        vgap = min(gaps)
        for i in range(len(horizontal_lines)-1):
            if horizontal_lines[i+1] - horizontal_lines[i] == hwidth:
                anchor_x = horizontal_lines[i]
                break
        for i in range(len(vertical_lines)-1):
            if vertical_lines[i+1] - vertical_lines[i] == vwidth:
                anchor_y = vertical_lines[i]
                break
        self.sqinfo = {
            'anchor_x':anchor_x,
            'anchor_y':anchor_y,
            'hwidth':hwidth,
            'vwidth':vwidth,
            'hgap':hgap,
            'vgap':vgap,
            'h':hgap+hwidth,
            'v':vgap+vwidth
        }
        print(f'左上角方块锚点坐标({anchor_x},{anchor_y})，参考值（20,137）')
        print(f'方块高度{hwidth}, 方块高度间隔{hgap}')
        print(f'方块宽度{vwidth}, 方块宽度间隔{vgap}')
        print('识别信息保存到sqinfo.json')
        print()
        json.dump(self.sqinfo, open('sqinfo.json','w'), indent=2)
        return self.sqinfo

    def crop_region(self, square):
        (x1, y1, x2, y2) = square
        # 通过切片提取矩形区域
        cropped_region = self.image[y1:y2, x1:x2]
        return cropped_region

    def get_matrix(self, image):
        self.image = image
        sqinfo = self.get_sqinfo(image)
        # self.squares = self.find_all_squares() # 寻找所有方块的四角坐标 (x1, y1, x2, y2)
        squares = []
        for i in range(16):
            for j in range(10):
                squares.append((sqinfo['anchor_x']+j*sqinfo['h'],
                                sqinfo['anchor_y']+i*sqinfo['v'],
                                sqinfo['anchor_x']+sqinfo['hwidth']+j*sqinfo['h'],
                                sqinfo['anchor_y']+sqinfo['vwidth']+i*sqinfo['v']))
        if len(squares)!= 160:
            print(squares)
            print('find squares error!')
            return None, squares
        self.crop_images = list(map(self.crop_region, squares)) # 根据坐标提取每个方块图片
        recognized_digits = list(map(recognize_digit, self.crop_images))  # 多线程识别图片
        self.digits_matrix = []
        for i in range(16):
            self.digits_matrix.append((recognized_digits[i * 10:i * 10 + 10]))
        return self.digits_matrix, squares



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
        """任意和为10的连续矩形，行优先的消除逻辑"""
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
        """任意和为10的连续矩形，列优先的消除逻辑"""
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

                # 搜索右边
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
                    else:
                        # 如果是0，继续向右查找
                        if self.cal_matrix[x, begin_y] == 0:
                            continue
                        else:
                            break

                # 搜索左边
                for x in range(begin_x - 1, -1, -1):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({x}, {begin_y})")
                        End = False
                        break
                    else:
                        # 如果是0，继续向左查找
                        if self.cal_matrix[x, begin_y] == 0:
                            continue
                        else:
                            break

                # 搜索下面
                for y in range(begin_y + 1, 10):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, begin_y] = 0
                        self.cal_matrix[begin_x, y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({begin_x}, {y})")
                        End = False
                        break
                    else:
                        # 如果是0，继续向下查找
                        if self.cal_matrix[begin_x, y] == 0:
                            continue
                        else:
                            break

                # 搜索上面
                for y in range(begin_y - 1, -1, -1):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, begin_y] = 0
                        self.cal_matrix[begin_x, y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({begin_x}, {y})")
                        End = False
                        break
                    else:
                        # 如果是0，继续向上查找
                        if self.cal_matrix[begin_x, y] == 0:
                            continue
                        else:
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

                # 搜索右边
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
                    else:
                        # 如果是0，继续向右查找
                        if self.cal_matrix[x, begin_y] == 0:
                            continue
                        else:
                            break

                # 搜索左边
                for x in range(begin_x - 1, -1, -1):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({x}, {begin_y})")
                        End = False
                        break
                    else:
                        # 如果是0，继续向左查找
                        if self.cal_matrix[x, begin_y] == 0:
                            continue
                        else:
                            break

                # 搜索下面
                for y in range(begin_y + 1, 10):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, begin_y] = 0
                        self.cal_matrix[begin_x, y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({begin_x}, {y})")
                        End = False
                        break
                    else:
                        # 如果是0，继续向下查找
                        if self.cal_matrix[begin_x, y] == 0:
                            continue
                        else:
                            break

                # 搜索上面
                for y in range(begin_y - 1, -1, -1):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, begin_y] = 0
                        self.cal_matrix[begin_x, y] = 0
                        if action:
                            self.actions.append(f"消除 ({begin_x}, {begin_y}) 和 ({begin_x}, {y})")
                        End = False
                        break
                    else:
                        # 如果是0，继续向上查找
                        if self.cal_matrix[begin_x, y] == 0:
                            continue
                        else:
                            break
        self.cal_two_y(End=End, action=action)

    def run_strategy(self, strategy, action=False):
        """按照策略执行多步骤操作"""
        self.cal_matrix = self.matrix.copy()
        if strategy[0] == 1:
            self.cal_two_x(action=action)
        elif strategy[0] == 2:
            self.cal_two_y(action=action)
        elif strategy[0] == 0 and strategy[1] != 0:
            pass  # 仅执行 strategy[1] 的操作

        if strategy[1] == 1:
            self.cal_all_x(action=action)
        elif strategy[1] == 2:
            self.cal_all_y(action=action)
        elif strategy[1] == 0 and strategy[0] != 0:
            pass  # 仅执行 strategy[0] 的操作


    def execute_strategy(self, strategy):
        """执行指定策略并返回分数"""
        self.actions.clear()
        self.run_strategy(strategy, action=True)
        return (strategy, self.score(), self.actions.copy())


if __name__ == "__main__":
    screenshot = cv2.imread("test11.png")

    # 识别数字矩阵
    recognizer = Recognizer()
    matrix, _ = recognizer.get_matrix(screenshot)

    # 将识别到的数字矩阵保存到 TXT 文件
    with open("matrix_output.txt", "w") as file:
        for row in matrix:
            file.write(" ".join(map(str, row)) + "\n")
    print("识别到的数字矩阵已保存到 matrix_output.txt")

    # 初始化 Eliminater
    eliminater = Eliminater(matrix)

    # 策略描述映射
    strategy_descriptions_first = {
        0: "不进行操作",
        1: "两位数和为10（行优先）",
        2: "两位数和为10（列优先）"
    }

    strategy_descriptions_second = {
        0: "不进行操作",
        1: "多位数和为10（行优先）",
        2: "多位数和为10（列优先）"
    }

    # 策略计算
    strategies = [
        [0, 1], [0, 2], [1, 0], [2, 0],
        [1, 1], [1, 2], [2, 1], [2, 2]
    ]

    strategy_scores = {}
    strategy_actions = {}

    for strategy in strategies:
        result = eliminater.execute_strategy(strategy)
        score = result[1]  # 策略得分
        actions = result[2]  # 策略执行的动作
        strategy_scores[str(strategy)] = score
        strategy_actions[str(strategy)] = actions

    # 找到最高分策略
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    best_score = strategy_scores[best_strategy]
    best_actions = strategy_actions[best_strategy]

    # 将字符串解析为列表
    parsed_strategy = list(map(int, best_strategy.strip('[]').split(',')))

    # 保存结果
    with open("result.txt", "w") as file:
        file.write(
            f"最佳策略: {best_strategy} "
            f"({strategy_descriptions_first[parsed_strategy[0]]}, {strategy_descriptions_second[parsed_strategy[1]]})\n"
        )
        file.write(f"得分: {best_score}\n")
        file.write("消除步骤:\n")
        file.write("\n".join(best_actions))

    print("结果已保存到 result.txt")
