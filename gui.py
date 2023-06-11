from process import extract_digit, find_puzzle
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import imutils
from sudoku import Sudoku
import pytesseract
from backtrack import solve, is_valid_board
import PySimpleGUI as sg
import os
import cv2
from PIL import Image

# Global variable
file_path = ""
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
config = r'--oem 3 --psm 6 outputbase digits'
model = load_model("model/ocrMODEL.h5")


class SudokuGUI:
    def __init__(self):
        self.entries = {}
        self.window = sg.Window("Sudoku Solver", self.create_layout(), finalize=True)

    def create_layout(self):
        layout = []
        for i in range(9):
            row = []
            for j in range(9):
                key = f'{i}_{j}'
                entry = sg.Input(size=(2, 1), key=key, pad=(5, 5), font=('Helvetica', 24), enable_events=True)
                row.append(entry)
                self.entries[key] = entry
            layout.append(row)
        layout.append([sg.Button('Solve', font=('Helvetica', 16), key='Solve', bind_return_key=True),
                       sg.Button('Next', font=('Helvetica', 16), key='Next', visible=False)])  # Add the 'Next' button
        return layout
    
    def display_board(self, board):  # Added method to display the board
        self.update_grid(board)
        self.window.refresh()

    def start(self):
        while True:
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED or event == 'Done':
                break

    def update_grid(self, board):
        for i in range(9):
            for j in range(9):
                key = f'{i}_{j}'
                value = board[i][j]
                if value > 0:  # Since we know that the values are integers, we can just check if it's greater than 0
                    self.window[key].update(value=str(value))
                else:
                    self.window[key].update(value="")

    def get_grid(self):  # Added method to retrieve grid values
        grid = []
        for i in range(9):
            row = []
            for j in range(9):
                key = f'{i}_{j}'
                value = self.window[key].get()  # Change here
                if value.isdigit():  # check if the entry is a digit
                    row.append(int(value))
                else:  # if it's not a digit, consider it as 0
                    row.append(0)
            grid.append(row)
        return grid


def browse_files():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    global file_path
    file_path = sg.popup_get_file("Select a File", initial_folder=script_dir,
                                  file_types=(("JPEG files", "*.jpg*"), ("all files", "*.*")))


def display_image(image_file):
    #image = Image.open(image_file)
    #image.show()
    image_file = ""


def display_solution(file_path, puzzleImage, cellLocs, board):
    for (cellRow, boardRow) in zip(cellLocs, board):
        # loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # unpack the cell coordinates
            startX, startY, endX, endY = box
            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            # determine the color based on the column index
            if (box[0] // (puzzleImage.shape[1] // 9)) % 2 == 0:
                color = (0, 0, 255)  # Red color for even columns
            else:
                color = (0, 255, 255)  # Yellow color for odd columns
            # draw the result digit on the Sudoku puzzle image
            cv2.putText(puzzleImage, str(digit), (textX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)



def convert_cv2_img_to_bytes(image):
    """Convert cv2 image to bytes for PySimpleGUI."""
    ret, png = cv2.imencode('.png', image)
    return png.tobytes()

def display_image_window(img_data, window_title='Image'):
    """Create a window to display an image."""
    layout = [[sg.Image(data=img_data, key='image')]]
    return sg.Window(window_title, layout, finalize=True, resizable=True)

def start_processing(selected_model):
    if file_path:
        board, puzzleImage, cellLocs = process_image(file_path, selected_model, model)

        # Create GUI
        gui = SudokuGUI()

        # Update the GUI with the OCR'd board
        gui.update_grid(board.tolist())

        # Display the initial puzzle image in a separate window
        puzzle_image_window = display_image_window(convert_cv2_img_to_bytes(puzzleImage), 'Original Puzzle')

        while True:
            event, values = gui.window.read(timeout=100)  # Add a timeout so other windows can be read
            if event == sg.WINDOW_CLOSED:
                puzzle_image_window.close()
                break
            elif event == 'Solve':
                # Get the user-validated sudoku
                validated_sudoku = np.array(gui.get_grid(), dtype='int')

                # Solve the sudoku
                print("[INFO] Solving Sudoku puzzle...")
                if is_valid_board(validated_sudoku):
                    solve(validated_sudoku)
                else:
                    print("[ERROR] The provided Sudoku board is invalid.")
                    continue

                # Update the GUI with the solved sudoku
                gui.update_grid(validated_sudoku.tolist())

                # Display the solution on the puzzle image
                display_solution(file_path, puzzleImage, cellLocs, validated_sudoku)

                # Update the puzzle image window with the solution image
                puzzle_image_window['image'].update(data=convert_cv2_img_to_bytes(puzzleImage))

                # Make the 'Next' button visible
                gui.window['Next'].update(visible=True)
            elif event == 'Next':
                gui.window.close()
                puzzle_image_window.close()
                break



def predict_with_pytesseract(digit):
    text = pytesseract.image_to_string(digit, config=config)
    if text.strip().isdigit():
        return int(text)
    return 0


def predict_with_model(digit, model):
    roi = cv2.resize(digit, (28, 28))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    prediction = model.predict(roi).argmax(axis=1)[0]
    return prediction


def process_image(image_file, selected_model, model):
    print("[INFO] Processing image...")
    image = cv2.imread(image_file)
    image = imutils.resize(image, width=600)
    (puzzleImage, warped) = find_puzzle(image, debug=1)
    board = np.zeros((9, 9), dtype="int")
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    cellLocs = []
    for y in range(0, 9):
        row = []
        for x in range(0, 9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            row.append((startX, startY, endX, endY))
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=0)
            if digit is not None:
                if selected_model == "pytesseract":
                    prediction = predict_with_pytesseract(digit)
                elif selected_model == "model":
                    prediction = predict_with_model(digit, model)
                board[y, x] = prediction
        cellLocs.append(row)
    print("[INFO] OCR'd Sudoku board:")
    puzzle = Sudoku(3, 3, board=board.tolist())
    puzzle.show()
    return board, puzzleImage, cellLocs


# GUI

def start_program():
    window = None
    while True:
        if window is None:
            # Create the main window when the program starts or when it's closed
            layout = [
                [sg.Button("Browse files", key='Browse')],
                [sg.Button("Start Processing", key='Start')],
                [sg.Radio("PyTesseract", "model", default=True, key='PyTesseract'), sg.Radio("Model", "model", key='Model')]
            ]
            window = sg.Window("Sudoku Solver", layout)

        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'Browse':
            browse_files()
        elif event == 'Start':
            selected_model = 'pytesseract' if values['PyTesseract'] else 'model'
            start_processing(selected_model)
            window = None  # Mark the main window to be restarted


if __name__ == "__main__":
    start_program()