import cv2
import numpy as np

VERTICAL = True
HORIZONTAL = False


# 이미지를 흑백처리한 후 threshold값을 정해 이진화를 진행한다.
def threshold(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # COLOR_BGR2GRAY = 6
    # image = cv2.cvtColor(image, 6, image)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return image


def gaussian_filter(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image


def weighted(value):
    standard = 10
    return int(value * (standard / 10))


def closing(image):
    kernel = np.ones((weighted(5), weighted(5)), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def put_text(image, text, loc):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(text), loc, font, 0.6, (255, 0, 0), 2)


def get_center(y, h):
    return (y + y + h) / 2      # same as y + (h / 2)


def get_line(image, axis, axis_value, start, end, length):  # 검출된 객체 좌표를 이용
    if axis:
        points = [(i, axis_value) for i in range(start, end)]
    else:
        points = [(axis_value, i) for i in range(start, end)]

    pixels = 0

    for i in range(len(points)):            # 1차원 배열일 경우 요소수를 반환
        (y, x) = points[i]
        pixels += (image[y][x] == 255)

        next_point = image[y + 1][x] if axis else image[y][x + 1]

        if next_point == 0 or i == len(points) - 1:
            if pixels >= weighted(length):
                break                   # 검출완료
            else:
                pixels = 0

    return y if axis else x, pixels


def stem_detection(image, stats, length):       # 객체 내에 존재하는 세로 직선을 검출하는 함수
    (x, y, w, h, area) = stats
    stems = []
    for col in range(x, x + w):                 # 하나의 객체(사각형)의 가로 범위
        end, pixels = get_line(image, VERTICAL, col, y, y + h, length)
        if pixels:      # 직선임이 검출되면.
            if len(stems) == 0 or abs(stems[-1][0] + stems[-1][2] - col) >= 1:
                # x좌표, w값 = 1,
                (x, y, w, h) = col, end - pixels + 1, 1, pixels
                stems.append([x, y, w, h])
            else:
                stems[-1][2] += 1       # 너비 값만 1 추가
    return stems


def count_rect_pixels(image, rect):
    x, y, w, h = rect
    pixels = 0
    for row in range(y, y + h):
        for col in range(x, x + w):
            if image[row][col] == 255:
                pixels += 1
    return pixels


def compo(image):
    value = []
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    for i in range(len(stats)):
        value = stats[i][4]

    return value


def count_pixels_part(image, area_top, area_bot, area_col):
    cnt = 0
    flag = False
    for row in range(area_top, area_bot):
        if not flag and image[row][area_col] == 255:
            flag = True
            cnt += 1
        elif flag and image[row][area_col] == 0:
            flag = False
    return cnt
#

