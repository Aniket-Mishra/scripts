import pyautogui
import timer_keypress
import time




if __name__ == '__main__':
    new_mail_location = pyautogui.locateOnScreen('screenshots/new_mail.png')

    print(new_mail_location)

    new_mail_point = pyautogui.center(new_mail_location)

    print(new_mail_point)

    new_mail_x = new_mail_point.x
    new_mail_y = new_mail_point.y

    # pyautogui.click(new_mail_x, new_mail_y)
    pyautogui.doubleClick('screenshots/new_mail.png')

    time.sleep(1)
    pyautogui.alert(text='Press Any key after closing this popup. Press and Hold Esc to end script.', title='Attention!', button='OK')
    # pyautogui.press('enter')

    timer_keypress.listener_func()
