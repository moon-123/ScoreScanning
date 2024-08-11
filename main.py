import cv2
import os
import functions as fs
import numpy as np
import modules
import pyfirmata
import time
import Arduino as ar

# board = pyfirmata.Arduino('COM3')
# servo = board.get_pin('d:11:s')  # 11번핀을 서보모터 신호선으로 설정

resource_path = os.getcwd() + "/res/"
image_0 = cv2.imread(resource_path + "music_sample.jpeg")
# image_0 = cv2.imread(resource_path + "scanned_sheet.jpeg")


image_1 = modules.remove_noise(image_0, 2)
height, width = image_1.shape
histogram = np.zeros(image_1.shape, np.uint8)

for row in range(height):
    pixels = 0
    for col in range(width):
        pixels += (image_1[row][col] == 255)  # 한 행에 존재하는 픽셀의 개수를 셈
    for pixel in range(pixels):
        histogram[row][pixel] = 255

image_1_resized = modules.image_crop(image_1)

image_2, staves = modules.remove_staves(image_1_resized)

image_3, staves = modules.normalization(image_2, staves, 10)

image_4, objects = modules.object_detection(image_3, staves)

image_5, objects = modules.object_analysis(image_4, objects)
print("== recognition ==")
image_6, key, beats, pitches = modules.recognition(image_5, staves, objects)

# arduino = ar.Arduino_Servo(board, beats, pitches, servo)

# arduino.test_led()

# arduino.test_servo()
# arduino.servo_with_pitch(pitches)
cv2.imwrite('res0.jpg', image_0)
cv2.imwrite('res1.jpg', image_1)
cv2.imwrite('res2.jpg', image_2)
cv2.imwrite('res3.jpg', image_3)
cv2.imwrite('res4.jpg', image_4)
cv2.imwrite('res5.jpg', image_5)
cv2.imwrite('res6.jpg', image_6)
cv2.imwrite('histogram.jpg', histogram)

# 이미지 출력
cv2.imshow('image', histogram)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

