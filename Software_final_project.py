import cv2
import numpy as np
import pickle
from collections import Counter
import json
import sys
import time

# Load the template
template = pickle.load(open('template.pkl', 'rb'))

def detect_digit(image):
    """
    Recognize a digit from the given image using template matching.

    Args:
        image (numpy.ndarray): The input image containing a digit.

    Returns:
        int: The recognized digit (0-9).
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_ = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    scores = np.zeros(10)
    for number, template_img in template.items():
        score = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
        scores[int(number)] = np.max(score)
    if np.max(scores) < 200000:
        print('Recognition error!')
    return np.argmax(scores)

class Recognizer:
    def __init__(self):
        """
        Initialize the Recognizer object and attempt to load recognition parameters from 'sqinfo.json'.
        """
        try:
            self.sqinfo = json.load(open('sqinfo.json', 'r'))
            print()
            print('Loaded recognition module from sqinfo.json')
            print(f"Top-left square anchor coordinates ({self.sqinfo['anchor_x']},{self.sqinfo['anchor_y']})")
            print(f"Square height {self.sqinfo['hwidth']}, height gap {self.sqinfo['hgap']}")
            print(f"Square width {self.sqinfo['vwidth']}, width gap {self.sqinfo['vgap']}")
            print()
            return
        except:
            pass

    def extract_square_info(self, image):
        """
        Calculate and return the recognition parameters from the given image.

        Args:
            image (numpy.ndarray): The input image to analyze.

        Returns:
            dict: A dictionary containing recognition parameters.
        """
        try:
            return self.sqinfo
        except:
            print()
            print('Initializing recognition module. Please verify if positioning is accurate.')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img1 = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(img1, 50, 150)
        # Use Hough Line Transform to detect lines
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
                # Categorize lines by angle; thresholds can be adjusted based on actual conditions
                if 0 <= int(theta * 180 / np.pi) <= 2 or 178 <= int(theta * 180 / np.pi) <= 182:
                    horizontal_lines.append(int(x0))
                elif 88 <= int(theta * 180 / np.pi) <= 92:
                    vertical_lines.append(int(y0))
        # Sort horizontal lines from top to bottom
        horizontal_lines.sort()
        vertical_lines.sort()
        gaps = []
        for i in range(len(horizontal_lines) - 1):
            gaps.append(horizontal_lines[i + 1] - horizontal_lines[i])
        cnt = Counter(gaps)
        gaps = [cnt.most_common(2)[0][0], cnt.most_common(2)[1][0]]
        hwidth = max(gaps)
        hgap = min(gaps)
        gaps = []
        for i in range(len(vertical_lines) - 1):
            gaps.append(vertical_lines[i + 1] - vertical_lines[i])
        cnt = Counter(gaps)
        gaps = [cnt.most_common(2)[0][0], cnt.most_common(2)[1][0]]
        vwidth = max(gaps)
        vgap = min(gaps)
        for i in range(len(horizontal_lines) - 1):
            if horizontal_lines[i + 1] - horizontal_lines[i] == hwidth:
                anchor_x = horizontal_lines[i]
                break
        for i in range(len(vertical_lines) - 1):
            if vertical_lines[i + 1] - vertical_lines[i] == vwidth:
                anchor_y = vertical_lines[i]
                break
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
        print(f'Top-left square anchor coordinates ({anchor_x},{anchor_y}), reference value (20,137)')
        print(f'Square height {hwidth}, height gap {hgap}')
        print(f'Square width {vwidth}, width gap {vgap}')
        print('Recognition information saved to sqinfo.json')
        print()
        json.dump(self.sqinfo, open('sqinfo.json', 'w'), indent=2)
        return self.sqinfo

    def extract_subimage(self, square):
        """
        Extract a rectangular region from the image based on the given coordinates.

        Args:
            square (tuple): A tuple of four integers (x1, y1, x2, y2) representing the coordinates of the region.

        Returns:
            numpy.ndarray: The cropped region of the image.
        """
        (x1, y1, x2, y2) = square
        # Extract the rectangular region using slicing
        cropped_region = self.image[y1:y2, x1:x2]
        return cropped_region

    def get_matrix(self, image):
        """
        Processes the given image to extract a matrix of digits.

        This function calculates the coordinates of all squares in a grid layout, crops the corresponding regions from the image,
        recognizes the digits in each square using a recognition function, and organizes the recognized digits into a 16x10 matrix.

        Args:
            image: The input image from which the matrix is to be extracted.

        Returns:
            tuple: A tuple containing:
                - digits_matrix (list of list of int): The 16x10 matrix of recognized digits.
                - squares (list of tuple): The list of coordinates for each square in the grid. Each tuple represents
                  the top-left and bottom-right corners of a square (x1, y1, x2, y2).
        """
        self.image = image
        sqinfo = self.extract_square_info(image)
        # squares: Find the coordinates of all squares (x1, y1, x2, y2)
        squares = []
        for i in range(16):
            for j in range(10):
                squares.append((sqinfo['anchor_x'] + j * sqinfo['h'],
                                sqinfo['anchor_y'] + i * sqinfo['v'],
                                sqinfo['anchor_x'] + sqinfo['hwidth'] + j * sqinfo['h'],
                                sqinfo['anchor_y'] + sqinfo['vwidth'] + i * sqinfo['v']))
        if len(squares) != 160:
            print(squares)
            print('find squares error!')
            return None, squares
        # Crop images from the identified square coordinates
        self.crop_images = list(map(self.extract_subimage, squares))
        # Recognize digits in the cropped images using a recognition function (multi-threaded)
        recognized_digits = list(map(detect_digit, self.crop_images))
        self.digits_matrix = []
        for i in range(16):
            self.digits_matrix.append((recognized_digits[i * 10:i * 10 + 10]))
        return self.digits_matrix, squares


class Eliminater:
    """
    Elimination Module responsible for operating on the matrix based on the given strategy.

    Attributes:
        matrix (np.array): The original matrix to operate on.
        cal_matrix (np.array): A copy of the matrix used for calculations.
        actions (list): A list to record elimination actions.
    """

    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.cal_matrix = self.matrix.copy()
        self.actions = []

    def score(self):
        """
        Calculate the current number of non-zero blocks remaining.

        Returns:
            int: The score calculated as 160 minus the count of non-zero blocks.
        """
        return 160 - np.sum(self.cal_matrix.astype(bool))

    def eliminate_all_rows(self, End=False, action=False):
        """
        Elimination logic for any continuous rectangle with a sum of 10, row-priority.

        Args:
            End (bool): Flag indicating whether to stop the recursion.
            action (bool): Whether to record actions.
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
                                self.actions.append(
                                    f"Eliminate ({begin_x}:{begin_x + x_len}, {begin_y}:{begin_y + y_len})")
                            End = False
        self.eliminate_all_rows(End=End, action=action)

    def eliminate_all_columns(self, End=False, action=False):
        """
        Elimination logic for any continuous rectangle with a sum of 10, column-priority.

        Args:
            End (bool): Flag indicating whether to stop the recursion.
            action (bool): Whether to record actions.
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
                                self.actions.append(
                                    f"Eliminate ({begin_x}:{begin_x + x_len}, {begin_y}:{begin_y + y_len})")
                            End = False
        self.eliminate_all_columns(End=End, action=action)

    def eliminate_pairs_rows(self, End=False, action=False):
        """
        Elimination logic for pairs of numbers whose sum is 10, row-priority.

        Args:
            End (bool): Flag indicating whether to stop the recursion.
            action (bool): Whether to record actions.
        """
        if End:
            return
        End = True
        for begin_x in range(0, 16):
            for begin_y in range(0, 10):
                if self.cal_matrix[begin_x, begin_y] == 0:
                    continue

                # Search to the right
                for x in range(begin_x + 1, 16):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"Eliminate ({begin_x}, {begin_y}) and ({x}, {begin_y})")
                        End = False
                        break
                    else:
                        break

                # Search to the left
                for x in range(begin_x - 1, -1, -1):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"Eliminate ({begin_x}, {begin_y}) and ({x}, {begin_y})")
                        End = False
                        break
                    else:
                        break

                # Search downwards
                for y in range(begin_y + 1, 10):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, begin_y] = 0
                        self.cal_matrix[begin_x, y] = 0
                        if action:
                            self.actions.append(f"Eliminate ({begin_x}, {begin_y}) and ({begin_x}, {y})")
                        End = False
                        break
                    else:
                        break

                # Search upwards
                for y in range(begin_y - 1, -1, -1):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, begin_y] = 0
                        self.cal_matrix[begin_x, y] = 0
                        if action:
                            self.actions.append(f"Eliminate ({begin_x}, {begin_y}) and ({begin_x}, {y})")
                        End = False
                        break
                    else:
                        break
        self.eliminate_pairs_rows(End=End, action=action)

    def eliminate_pairs_columns(self, End=False, action=False):
        """
        Elimination logic for pairs of numbers whose sum is 10, column-priority.

        Args:
            End (bool): Flag indicating whether to stop the recursion.
            action (bool): Whether to record actions.
        """
        if End:
            return
        End = True
        for begin_y in range(0, 10):
            for begin_x in range(0, 16):
                if self.cal_matrix[begin_x, begin_y] == 0:
                    continue

                # Search to the right
                for x in range(begin_x + 1, 16):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"Eliminate ({begin_x}, {begin_y}) and ({x}, {begin_y})")
                        End = False
                        break
                    else:
                        break

                # Search to the left
                for x in range(begin_x - 1, -1, -1):
                    if self.cal_matrix[x, begin_y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[x, begin_y] == 10:
                        self.cal_matrix[x, begin_y] = 0
                        self.cal_matrix[begin_x, begin_y] = 0
                        if action:
                            self.actions.append(f"Eliminate ({begin_x}, {begin_y}) and ({x}, {begin_y})")
                        End = False
                        break
                    else:
                        break

                # Search downwards
                for y in range(begin_y + 1, 10):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, begin_y] = 0
                        self.cal_matrix[begin_x, y] = 0
                        if action:
                            self.actions.append(f"Eliminate ({begin_x}, {begin_y}) and ({begin_x}, {y})")
                        End = False
                        break
                    else:
                        break

                # Search upwards
                for y in range(begin_y - 1, -1, -1):
                    if self.cal_matrix[begin_x, y] == 0:
                        continue
                    elif self.cal_matrix[begin_x, begin_y] + self.cal_matrix[begin_x, y] == 10:
                        self.cal_matrix[begin_x, begin_y] = 0
                        self.cal_matrix[begin_x, y] = 0
                        if action:
                            self.actions.append(f"Eliminate ({begin_x}, {begin_y}) and ({begin_x}, {y})")
                        End = False
                        break
                    else:
                        break
        self.eliminate_pairs_columns(End=End, action=action)

    def run_strategy(self, strategy, action=False):
        """
        Execute multiple steps according to the strategy.

        Args:
            strategy (list): A list indicating the sequence of operations.
            action (bool): Whether to record actions.
        """
        self.cal_matrix = self.matrix.copy()
        if strategy[0] == 1:
            self.eliminate_pairs_rows(action=action)
        elif strategy[0] == 2:
            self.eliminate_pairs_columns(action=action)
        elif strategy[0] == 3:
            self.eliminate_all_rows(action=action)
        elif strategy[0] == 4:
            self.eliminate_all_columns(action=action)
        elif strategy[0] == 0 and strategy[1] != 0:
            pass  # Execute only strategy[1]

        if strategy[1] == 1:
            self.eliminate_pairs_rows(action=action)
        elif strategy[1] == 2:
            self.eliminate_pairs_columns(action=action)
        elif strategy[1] == 3:
            self.eliminate_all_rows(action=action)
        elif strategy[1] == 4:
            self.eliminate_all_columns(action=action)
        elif strategy[1] == 0 and strategy[0] != 0:
            pass  # Execute only strategy[0]

    def execute_strategy(self, strategy):
        """
        Execute the specified strategy and return the score.

        Args:
            strategy (list): A list indicating the sequence of operations.

        Returns:
            tuple: The strategy, resulting score, and recorded actions.
        """
        self.actions.clear()
        self.run_strategy(strategy, action=True)
        return (strategy, self.score(), self.actions.copy())


if __name__ == "__main__":
    """
    Main script to process a matrix from an image, execute elimination strategies,
    and save results to a file. The script performs the following tasks:
    1. Reads an image and recognizes a numeric matrix.
    2. Saves the recognized matrix to a text file.
    3. Initializes the Eliminater with the matrix.
    4. Executes multiple elimination strategies.
    5. Finds and saves the best strategy based on scores.
    """

    screenshot = cv2.imread("test11.png")

    # Recognize the numeric matrix
    recognizer = Recognizer()
    matrix, _ = recognizer.get_matrix(screenshot)

    # Save the recognized matrix to a TXT file
    with open("matrix_output.txt", "w") as file:
        for row in matrix:
            file.write(" ".join(map(str, row)) + "\n")
    print("The recognized numeric matrix has been saved to matrix_output.txt")

    # Initialize Eliminater
    eliminater = Eliminater(matrix)

    # Strategy description mappings
    strategy_descriptions_first = {
        0: "No operation",
        1: "Two-digit sum equals 10 (row priority)",
        2: "Two-digit sum equals 10 (column priority)",
        3: "Multi-digit sum equals 10 (row priority)",
        4: "Multi-digit sum equals 10 (column priority)"
    }

    strategy_descriptions_second = {
        0: "No operation",
        1: "Two-digit sum equals 10 (row priority)",
        2: "Two-digit sum equals 10 (column priority)",
        3: "Multi-digit sum equals 10 (row priority)",
        4: "Multi-digit sum equals 10 (column priority)"
    }

    # Strategy calculations
    strategies = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 0], [2, 0], [3, 0], [4, 0],
        [1, 1], [1, 2], [1, 3], [1, 4],
        [2, 1], [2, 2], [2, 3], [2, 4],
        [3, 1], [3, 2], [3, 3], [3, 4],
        [4, 1], [4, 2], [4, 3], [4, 4]
    ]

    strategy_scores = {}
    strategy_actions = {}

    for strategy in strategies:
        result = eliminater.execute_strategy(strategy)
        score = result[1]  # Strategy score
        actions = result[2]  # Actions performed during the strategy execution
        strategy_scores[str(strategy)] = score
        strategy_actions[str(strategy)] = actions

    # Find the highest-scoring strategy
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    best_score = strategy_scores[best_strategy]
    best_actions = strategy_actions[best_strategy]

    # Parse the strategy string into a list
    parsed_strategy = list(map(int, best_strategy.strip('[]').split(',')))

    # Save results to a file
    with open("result.txt", "w") as file:
        file.write(
            f"Best Strategy: {best_strategy} "
            f"({strategy_descriptions_first[parsed_strategy[0]]}, {strategy_descriptions_second[parsed_strategy[1]]})\n"
        )
        file.write(f"Score: {best_score}\n")
        file.write("Elimination Steps:\n")
        file.write("\n".join(best_actions))

    print("The results have been saved to result.txt")
