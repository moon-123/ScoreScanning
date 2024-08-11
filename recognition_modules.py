import functions as fs
import cv2


def recognize_key(image, staves, stats):
    (x, y, w, h, area) = stats
    fs.put_text(image, w, (x, y + h + fs.weighted(40)))
    fs.put_text(image, h, (x, y + h + fs.weighted(60)))
    # print(stats)
    top_condition = staves[0] + fs.weighted(5) >= y >= staves[0] - fs.weighted(5)
    bot_condition = staves[4] + fs.weighted(5) >= y + h >= staves[4] - fs.weighted(5)
    cen_condition = staves[2] + fs.weighted(5) >= fs.get_center(y, h) >= staves[2] - fs.weighted(5)
    width_condition = fs.weighted(18) >= w >= fs.weighted(10)
    height_condition = fs.weighted(45) >= h >= fs.weighted(35)
    '''
    print("top: " + str(top_condition))
    print("bot: " + str(bot_condition))
    print("cen: " + str(cen_condition))
    print("width: " + str(width_condition))
    print("height: " + str(height_condition))
'''
    ts_conditions = (
            top_condition and
            bot_condition and
            cen_condition and
            width_condition and
            height_condition
    )

    # print(ts_conditions)
    if ts_conditions:   # 박자표이다.
        return True, 0

    else:
        stems = fs.stem_detection(image, stats, 20)
        if stems[0][0] - x >= fs.weighted(3):       # 나중에 검출되면 -> 기둥의 너비를 고려해서 가중이 3으로 둠.
            key = int(10 * len(stems) / 2)          # 이건 #이다 라고 알려줌, 세로 직선이 두개니까 나누기 2
        else:
            key = 100 * len(stems)                  # 플랫

    return False, key


def recognize_note(image, staff, stats, stems, direction):
    x, y, w, h, area = stats
    notes = []
    pitches = []
    note_condition = (                  # 음표가 되기 위한 최소조건
        len(stems) and                  # 기둥의 유무
        w >= fs.weighted(10) and
        h >= fs.weighted(35) and
        area >= fs.weighted(95)
    )
    # print('==recognize_note==')
    if note_condition:                  # 조건에 맞으면
        for i in range(len(stems)):     # 기둥 하나당 머리 하나를 탐색
            stem = stems[i]
            head_exist, head_fill, head_center = recognize_note_head(image, stem, direction)
            if head_exist:
                tail_cnt = recognize_note_tail(image, i, stem, direction)
                dot_exist = recognize_note_dot(image, stem, direction, len(stems), tail_cnt)

                # 분류될 note의 종류는 총 10개
                # 머리만 존재하는 음표는 제외
                # 기둥과 머리는 필수조건, 꼬리 개수, 점 유무 선택조건
                # 점이 있으면 음수값
                # 2분음표, 4분음표, 8분음표, 16분음표, 32분음표

                note_classification = (
                    ((not head_fill and tail_cnt == 0 and not dot_exist), 2),
                    ((not head_fill and tail_cnt == 0 and dot_exist), -2),
                    ((head_fill and tail_cnt == 0 and not dot_exist), 4),
                    ((head_fill and tail_cnt == 0 and dot_exist), -4),
                    ((head_fill and tail_cnt == 1 and not dot_exist), 8),
                    ((head_fill and tail_cnt == 1 and dot_exist), -8),
                    ((head_fill and tail_cnt == 2 and not dot_exist), 16),
                    ((head_fill and tail_cnt == 2 and dot_exist), -16),
                    ((head_fill and tail_cnt == 3 and not dot_exist), 32),
                    ((head_fill and tail_cnt == 3 and dot_exist), -32)
                )

                for j in range(len(note_classification)):
                    # print(note_classification)
                    if note_classification[j][0]:
                        note = note_classification[j][1]
                        pitch = recognize_pitch(image, staff, head_center)
                        notes.append(note)      # 2, -2, 4, -4, ...
                        pitches.append(pitch)
                        # fs.put_text(image, note, (stem[0] - fs.weighted(10), stem[1] + stem[3] + fs.weighted(30)))
                        break
    return notes, pitches


def recognize_note_head(image, stem, direction):    # 기둥은 머리의 중간정도까지 연결되어있음
    (x, y, w, h) = stem     # 머리를 탐색하기 위한 구간을 정하는 과정. 구간이 명확하기 때문에 가능
    if direction:   # 기둥이 오른쪽에 존재, 머리가 왼쪽 아래 존재
        area_top = y + h - fs.weighted(7)
        area_bot = y + h + fs.weighted(7)
        area_left = x - fs.weighted(14)
        area_right = x
    else:           # 기둥이 왼쪽에 존재, 머리가 오른쪽 위에 존재
        area_top = y - fs.weighted(7)
        area_bot = y + fs.weighted(7)
        area_left = x + w
        area_right = x + w + fs.weighted(14)

    cv2.rectangle(image, (area_left, area_top, area_right - area_left, area_bot - area_top), (255, 0, 0), 1)

    cnt = 0     # 끊기지 않고 이어져 있는 선의 개수를 센다. => 특징중 하나.
    cnt_max = 0
    head_center = 0
    pixel_cnt = fs.count_rect_pixels(image, (area_left, area_top, area_right - area_left, area_bot - area_top))
    # 머리 객체 안의 픽셀 수 세기

    for row in range(area_top, area_bot):       # 머리가 존재하는 영역 맨 위부터 맨 아래까지
        col, pixels = fs.get_line(image, fs.HORIZONTAL, row, area_left, area_right, 5)
        # 길이가 5보다 큰 직선 모두 검출. 검출시 해당 직선의 최대 x 좌표 => col
        # 검출된 직선의 맨 끝 값 col, 길이 pixels

        pixels += 1     # pixels 1 증가? =>
        if pixels >= fs.weighted(5):      # pixels가 5보다 크면, 선이 존재한다면, 끊기지 않았다면.
            cnt += 1    # 직선이 하나 존재한다고 판단. 개수 1 증가
            cnt_max = max(cnt_max, pixels)  # 픽셀 개수가 가장 많을 때의 값을 저장, 가장 긴 직선의 길이 저장?
            head_center += row

    head_exist = (cnt >= 3 and pixel_cnt >= 50)
    head_fill = (cnt >= 8 and cnt_max >= 9 and pixel_cnt >= 80)
    head_center /= cnt

    return head_exist, head_fill, head_center


def recognize_note_tail(image, index, stem, direction):
    (x, y, w, h) = stem
    if direction:   # 꼬리가 오른쪽 위일 경우
        area_top = y
        area_bot = y + h - fs.weighted(15)  # 맨 아래에서 적당히 위 => 예상되는 꼬리의 하단 위치

        area_left = x + w
        area_right = x + w + fs.weighted(10)

    else:           # 꼬리가 오른쪽 아래일 경우
        area_top = y + fs.weighted(15)      # 맨 위에서 적당히 아래
        area_bot = y + h
        area_left = x + w
        area_right = x + w + fs.weighted(10)

    if index:       # ?
        area_col = x - fs.weighted(4)       # 기둥의 앞과 뒤 탐색
    else:
        area_col = x + w + fs.weighted(4)

    cnt = fs.count_pixels_part(image, area_top, area_bot, area_col)

    return cnt


def recognize_note_dot(image, stem, direction, tail_cnt, stems_cnt):
    (x, y, w, h) = stem
    if direction:
        if direction:
            area_top = y + h - fs.weighted(10)
            area_bot = y + h + fs.weighted(5)
            area_left = x + w + fs.weighted(2)
            area_right = x + w + fs.weighted(12)
        else:
            area_top = y - fs.weighted(10)
            area_bot = y + fs.weighted(5)
            area_left = x + w + fs.weighted(14)
            area_right = x + w + fs.weighted(24)

        dot_rect = (
            area_left,
            area_top,
            area_right - area_left,
            area_bot - area_top
        )

        pixels = fs.count_rect_pixels(image, dot_rect)

        threshold = (10, 15, 20, 30)        # 꼬리 개수에 따라 threshold 값 다르게
        if direction and stems_cnt == 1:
            return pixels >= fs.weighted(threshold[tail_cnt])   # 꼬리가 많으면 더 정교하게 확인
        else:
            return pixels >= fs.weighted(threshold[0])


def recognize_pitch(image, staff, head_center):
    pitch_lines = [staff[4] + fs.weighted(30) - fs.weighted(5) * i for i in range(21)]
    # 오선 간격을 10으로 설정하였기 때문에 선과 선 사이의 공간도 좌표라고 설정하려면 5씩 간격을 둬야함.
    # 총 1부터 21까지 21개의 좌표 생성
    for i in range(len(pitch_lines)):   # 21 반복
        line = pitch_lines[i]
        if line + fs.weighted(2) >= head_center >= line - fs.weighted(2):   # 머리가 어떤 좌표에 속하는지 넉넉하게 판단.

            return i
    # 반환값 i에는 21개의 좌표중 하나의 값이 저장됨


def recognize_rest(image, staff, stats):
    (x, y, w, h, area) = stats
    rest = 0
    center = fs.get_center(y, h)
    rest_condition = staff[3] > center > staff[1]   # 쉼표의 위치를 특정함
    if rest_condition:
        cnt = fs.count_pixels_part(image, y, y + h, x + fs.weighted(1))

        if fs.weighted(35) >= h >= fs.weighted(25):
            if cnt == 3 and fs.weighted(11) >= w >= fs.weighted(7):
                fs.put_text(image, "r4", (x, y + h + fs.weighted(30)))

            elif cnt == 1 and fs.weighted(14) >= w >= fs.weighted(11):
                fs.put_text(image, "r16", (x, y + h + fs.weighted(30)))

        elif fs.weighted(22) >= h >= fs.weighted(16):
            if fs.weighted(15) >= w >= fs.weighted(9):
                fs.put_text(image, "r8", (x, y + h + fs.weighted(30)))

            elif fs.weighted(8) >= h:
                if staff[1] + fs.weighted(5) >= center >= staff[1]:
                    fs.put_text(image, "r1", (x, y + h + fs.weighted(30)))

                elif staff[2] >= center >= staff[1] + fs.weighted(5):
                    fs.put_text(image, "r2", (x, y + h + fs.weighted(30)))

            if recognize_rest_dot(image, stats):
                rest *= -1
    return rest


def recognize_rest_dot(image, stats):
    (x, y, w, h, area) = stats
    area_top = y - fs.weighted(10)
    area_bot = y + fs.weighted(10)
    area_left = x + w
    area_right = x + w + fs.weighted(10)
    dot_rect = (
        area_left,
        area_top,
        area_right - area_left,
        area_bot - area_top
    )

    pixels = fs.count_rect_pixels(image, dot_rect)
    # fs.put_text(image, pixels, (x, y + h + fs.weighted(60)))
    # cv2.rectangle(image, dot_rect, (255, 0, 0) , 1)

    return pixels >= fs.weighted(10)


def recognize_whole_note(image, staff, stats):
    whole_note = 0
    pitch = 0
    (x, y, w, h, area) = stats
    while_note_condition = (
        fs.weighted(22) >= w >= fs.weighted(12) >= h >= fs.weighted(9)
    )
    if while_note_condition:
        dot_rect = (
            x + w,
            y - fs.weighted(10),
            fs.weighted(10),
            fs.weighted(20)
        )
        pixels = fs.count_rect_pixels(image, dot_rect)
        whole_note = -1 if pixels >= fs.weighted(10) else 1
        pitch = recognize_pitch(image, staff, fs.get_center(y, h))

    return whole_note, pitch