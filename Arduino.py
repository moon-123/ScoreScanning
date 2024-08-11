import pyfirmata
import time
import ctypes
from serial import SerialException

# try:
#     # Your code that may raise a SerialException here
#     # ...
#
# except SerialException:
#     win_error = ctypes.GetLastError()  # Get the Windows error code
#     error_message = "WriteFile failed (Error code: {})".format(win_error)
#     raise SerialException(error_message)

class Arduino_Servo:
    def __init__(self, board, note, pitches, servo):
        self.board = board
        self.note = note
        self.pitches = pitches
        self.servo = servo

    def move_servo(self, v, delay):
        self.servo.write(v)
        self.board.pass_time(delay)

    def center(self, delay):
        print('center')
        self.servo.write(90)
        self.board.pass_time(delay)

    def move_right(self, delay):
        print('move_right')
        self.servo.write(0)
        self.board.pass_time(delay)

    def move_left(self, delay):
        print('move_left')
        self.servo.write(180)
        self.board.pass_time(delay)

    def test_led(self):
        while None in self.pitches:
            self.pitches.remove(None)

        print(self.pitches)
        for i in self.pitches:
            print(f'pitches[i]: {i}')
            if i in [4, 5]:
                self.board.digital[2].write(1)
                time.sleep(1)
                self.board.digital[2].write(0)

            elif i in [6, 7]:
                self.board.digital[3].write(1)
                time.sleep(1)
                self.board.digital[3].write(0)

            elif i in [8, 9]:
                self.board.digital[4].write(1)
                time.sleep(1)
                self.board.digital[4].write(0)

            elif i in [10, 11]:
                self.board.digital[5].write(1)
                time.sleep(1)
                self.board.digital[5].write(0)

            elif i is None:
                continue

    def test_servo(self):
        DELAY = 1
        MIN = 5
        MAX = 175
        MID = 90
        self.center(DELAY)
        self.move_right(DELAY)
        self.move_left(DELAY)
        self.board.exit()

    def servo_with_pitch(self, pitches):
        try:
            for i in range(len(pitches)):
                print(f'pitches: {pitches[i]}')
                if pitches[i] == 4:
                    self.move_left(0.5)
                    self.move_right(0.5)
                else:
                    self.center(1)

        except SerialException:
            win_error = ctypes.GetLastError()  # Get the Windows error code
            error_message = "WriteFile failed (Error code: {})".format(win_error)
            raise SerialException(error_message)

