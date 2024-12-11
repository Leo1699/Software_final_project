# Optimal Path Solution for Sum to Ten Game

This project includes a core strategy module for a digital elimination game, designed to automatically recognize numbers on a game board and execute elimination strategies to optimize scores. The module encompasses functionalities such as number recognition, strategy execution, and strategy performance evaluation.

## Game Rules

In this game, there are a total of 160 digits, each ranging from 1 to 9. The objective is to eliminate all adjacent digit combinations that sum to 10 within a set timeframe. Each digit eliminated scores one point. For example, adjacent digits 1 and 9 can be eliminated, as can a sequence like 1, 2, 3, and 4 that adds up to 10.
When a digit is eliminated, it becomes blank, allowing new combinations to form. For instance, if a 5 is eliminated from a neighboring set of 4, 5, and 6, then 4 and 6 can be subsequently combined and eliminated. This dynamic aspect of the game allows for continuously evolving play as players strive to clear the board and maximize their score.

## Features

- **Number Recognition**: Automatically identify numbers from a provided game board image.
- **Strategy Execution**: Implement different strategies to eliminate number combinations that sum up to 10 on the game board.
- **Strategy Evaluation**: After executing strategies, evaluate each based on the game score and record the best score and elimination path.

## Dependencies

This project relies on the following Python libraries:

- OpenCV (`cv2`): Used for image processing.
- NumPy: Used for efficient numerical computations.
- Pickle: Used for loading and storing pre-trained templates.
- JSON: Used for loading and saving configuration files.

- `Software_final_project.py`: Main program file containing code for number recognition and strategy execution.
- `Notebook_Version.ipynb`: Notebook version of the main program.
- `template.pkl`: Stores pre-trained number templates.
- `sqinfo.json`: Stores information about the game board configuration. This can be directly used for the screenshot we provide. If the screenshot is taken with different size, generate a new one from the code.
- `screenshot`: Sample screenshots took from the game. Input of this project.
- `matrix_output`: Stores matrix for the input image. Generated from the code.
- `result.txt`: Final output. Stores best game score and elimination path.

## How to Run
With all dependencies, run `Software_final_project.py` with any picture from `screenshot` and `template.pkl` as input. 
