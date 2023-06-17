import cv2
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np


def convert_to_gray_and_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    return blurred


def apply_threshold(blurred):
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.bitwise_not(thresh)


def find_contours(thresh):
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return sorted(cnts, key=cv2.contourArea, reverse=True)


def find_puzzle_contour(cnts):
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    raise Exception(("Could not find Sudoku puzzle outline."))


def apply_perspective_transform(image, gray, puzzleCnt):
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    return puzzle, warped


def find_puzzle(image, debug=False):
    gray = convert_to_gray_and_blur(image)
    thresh = apply_threshold(gray)

    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    cnts = find_contours(thresh)
    puzzleCnt = find_puzzle_contour(cnts)

    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    return apply_perspective_transform(image, gray, puzzleCnt)


def apply_threshold_to_cell(cell):
    return cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def find_cell_contours(thresh):
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(cnts)


def get_mask(thresh, cnts):
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    return mask


def extract_digit(cell, debug=False):
    thresh = clear_border(apply_threshold_to_cell(cell))

    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    cnts = find_cell_contours(thresh)

    if len(cnts) == 0:
        return None

    mask = get_mask(thresh, cnts)

    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    if percentFilled < 0.02:
        return None

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)

    return digit
