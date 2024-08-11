import cv2
import numpy as np
import functions as fs
import recognition_modules as rs

# 이미지 이진화
# 0으로 가득찬 마스크 이미지이기 때문에 검은색임
# print(cnt)
# print(stats)

def rotate_image(image, degree):
    h, w = image.shape[:2]
    cx, cy = w // 1.9, h // 2.5
    m = cv2.getRotationMatrix2D((cx, cy), degree, 1.0)
    image = cv2.warpAffine(image, m, (h, w))
    return image


def line_detection(image):
    h, w = image.shape[:2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    edged_image = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edged_image, 1, np.pi/180, threshold=10, minLineLength=100, maxLineGap=10)
    # lines = cv2.HoughLines(edged_image, rho=1, theta=np.pi/180, threshold=200)

#    for line in lines:
#        x1, y1, x2, y2 = line[0]
#        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return lines


def separate_lines(lines, threshold):
    """
    검출된 라인들을 수평 라인과 수직 라인으로 분리합니다.
    """
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < threshold:
            # 수직 라인
            vertical_lines.append(line)
        elif abs(y2 - y1) < threshold:
            # 수평 라인
            horizontal_lines.append(line)
    return horizontal_lines, vertical_lines


def find_staff_lines(horizontal_lines, threshold):
    """
    수평 라인들 중 오선을 찾아내어 반환합니다.
    """
    staff_lines = []
    for i in range(len(horizontal_lines)):
        for j in range(i+1, len(horizontal_lines)):
            if abs(horizontal_lines[i][0][1] - horizontal_lines[j][0][1]) < threshold:
                # 두 수평 라인 사이의 거리가 threshold 이하면 오선으로 판단
                staff_lines.append((horizontal_lines[i], horizontal_lines[j]))
    return staff_lines


def remove_dot(image, w_size, h_size):
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    for i in range(1, cnt):         # 0번 객체는 배경이라 제외
        x, y, w, h, area = stats[i]
        if w < w_size | h < h_size:
            # cv2.rectangle(image, (x, y, w, h), (255, 0, 0), 1)
            image[y:y+h, x:x+w] = 0
    return image


def component_detection(image):
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    print(cnt)
    for i in range(1, cnt):         # 0번 객체는 배경이라 제외
        x, y, w, h, area = stats[i]
        if w > image.shape[1] * 0.5:
            cv2.rectangle(image, (x, y, w, h), (255, 0, 0), 1)

    return image


def staff_detection(image):
    staff_stats = []
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    print(cnt)
    for i in range(1, cnt):         # 0번 객체는 배경이라 제외
        x, y, w, h, area = stats[i]
        if w > image.shape[1] * 0.5:
            staff_stats.append((x, y, w, h, area))
    print(staff_stats)
    return image


def find_vertex(image, staff_stats):

    return image


#######################################################################################################################
# 지금 사용하는 함수

def image_crop(image):

    h, w = image.shape
    lower_col = 0
    for row in range(w):
        for col in range(h):
            if image[col, row] == 255:
                lower_col = col

    cropped_image = image[0: lower_col + 200, 0: w]

    return cropped_image


def remove_noise(image, component): # component는 배경 생
    print("== remove_noise ==")
    image = fs.threshold(image)
    mask = np.zeros(image.shape, np.uint8)  # uint8 부호없는 8비트 정수, 0 ~ 255, 메모리 사용이 비교적 적어 영상처리에 많이 사용
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    for i in range(component, cnt):         # 0번 객체는 배경이라 제외
        x, y, w, h, area = stats[i]
        if w > image.shape[1] * 0.5:
            cv2.rectangle(mask, (x, y, w, h), (255, 0, 0), -1)

    masked_image = cv2.bitwise_and(image, mask)
    # cv2.imshow("res", masked_image)

    return masked_image


def remove_staves(image):
    print("== remove_staves ==")
    height, width = image.shape
    staves = []

    for row in range(height):
        pixels = 0
        for col in range(width):
            pixels += (image[row][col] == 255)
        if pixels >= width * 0.5:
            if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:
                staves.append([row, 0])     # idx 0 => row, idx 1 => 0
                # print(pixels)
            else:
                # print("-")
                staves[-1][1] += 1      # idx 1 => +1

    for staff in range(len(staves)):
        top_pixel = staves[staff][0]
        bot_pixel = staves[staff][0] + staves[staff][1]

        for col in range(width):
            if image[top_pixel - 1][col] == 0 and image[bot_pixel + 1][col] == 0:
                for row in range(top_pixel, bot_pixel + 1):
                    image[row][col] = 0
    # print(staves)
    return image, [x[0] for x in staves]


def normalization(image, staves, standard):
    print("== normalization ==")
    avg_distance = 0
    lines = int(len(staves) / 5)  # 보표의 개수
    for line in range(lines):
        for staff in range(4):
            staff_above = staves[line * 5 + staff]
            staff_below = staves[line * 5 + staff + 1]
            avg_distance += abs(staff_above - staff_below)  # 오선의 간격을 누적해서 더해줌
    avg_distance /= len(staves) - lines  # 오선 간의 평균 간격

    height, width = image.shape  # 이미지의 높이와 넓이
    weight = standard / (avg_distance + 1)  # 기준으로 정한 오선 간격을 이용해 가중치를 구함
    new_width = int(width * weight)  # 이미지의 넓이에 가중치를 곱해줌
    new_height = int(height * weight)  # 이미지의 높이에 가중치를 곱해줌

    image = cv2.resize(image, (new_width, new_height))  # 이미지 리사이징
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 이미지 이진화
    staves = [x * weight for x in staves]  # 오선 좌표에도 가중치를 곱해줌

    return image, staves


def object_detection(image, staves):
    print("== object_detection ==")
    lines = int(len(staves) / 5)
    objects = []

    closing_image = fs.closing(image)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_image)
    for i in range(1, cnt):
        (x, y, w, h, area) = stats[i]
        if w >= fs.weighted(3):
            center = fs.get_center(y, h)
            for line in range(lines):
                area_top = staves[line * 5] - fs.weighted(20)
                area_bot = staves[(line + 1) * 5 - 1] + fs.weighted(20)

                if area_top <= center <= area_bot:
                    objects.append([line, (x, y, w, h, area)])

    objects.sort()

    return image, objects


def object_analysis(image, objects):
    print("== object_analysis ==")
    for obj in objects:
        stats = obj[1]
        stems = fs.stem_detection(image, stats, 30)  # 이미지, stats는 객체 정보, 30은 검출할 직선 길이.
        direction = None
        if len(stems) > 0:  # 기둥이 검출이 되었나
            if stems[0][0] - stats[0] >= fs.weighted(5):  # 5의 간격 보다 크냐? -> 객체 시작좌표보다 뒤에 직선이 있을 경우
                direction = True
            else:
                direction = False
        obj.append(stems)
        obj.append(direction)

    return image, objects


def recognition(image, staves, objects):
    key = 0
    time_signature = False
    beats = []
    pitches = []

    for i in range(1, len(objects)):
        obj = objects[i]
        line = obj[0]
        stats = obj[1]
        stems = obj[2]
        direction = obj[3]
        (x, y, w, h, area) = stats
        staff = staves[line * 5: (line + 1) * 5]

        if not time_signature:
            ts, temp_key = rs.recognize_key(image, staff, stats)
            time_signature = ts                 # 박자표를 찾을 때까지 False => 계속 key값 반환
            key += temp_key                     # #이면 10, b이면 100

        else:
            notes = rs.recognize_note(image, staff, stats, stems, direction)
            # (notes, pitches) 반환
            if len(notes[0]):   # 0번 인덱스인 notes가 인식되면
                for beat in notes[0]:   # notes[0]은 2, -2, 4, -4 .. 등의 값
                    beats.append(beat)      # beat 에 4, 4, 4, 4, ... 값이 저장.
                for pitch in notes[1]:  # notes[1]은 1 ~ 21
                    pitches.append(pitch)
            else:
                rest = rs.recognize_rest(image, staff, stats)
                if rest:
                    beats.append(rest)
                    pitches.append(-1)
                else:
                    whole_note, pitch = rs.recognize_whole_note(image, staff, stats)
                    if whole_note:
                        beats.append(whole_note)
                        pitches.append(pitch)

        cv2.rectangle(image, (x, y, w, h), (255, 0, 0), 1)
        fs.put_text(image, i, (x, y - fs.weighted(20)))

    print(beats)
    print(pitches)
    return image, key, beats, pitches
#######################################################################################################################

