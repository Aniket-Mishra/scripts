"""
How to use:

1. Open the application that logs you out cuz of inactivity.

2. click on text widow in the application.

3. run the script.
    >>> python timer_keypress.py

EZPZ.

4. Press esc to stop the script.

5. To change timer, go to line 34 and set after how long you want the keys to be pressed in seconds.

Have fun!
"""

import sys
import time

from pynput.keyboard import Controller, Key, Listener

## Global objects
keyboard = Controller()
start_time = time.time()  # system time at execution


def on_key_press(key):
    if key == Key.esc:
        print("Esc was pressed.")
        # return False #stop listener
        sys.exit()

    print("{} pressed".format(key))
    keyboard.press(Key.space)
    time.sleep(120.0 - ((time.time() - start_time) % 120.0))


def on_key_release(key):
    print("{} released".format(key))
    keyboard.release(Key.space)


def listener_func():
    with Listener(
        on_press=on_key_press, on_release=on_key_release
    ) as listener:
        listener.join()
