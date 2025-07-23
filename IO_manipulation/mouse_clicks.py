import pyautogui

print(pyautogui.size())

print(pyautogui.position())


pyautogui.click(pyautogui.position())

pyautogui.scroll(200)

pyautogui.typewrite("hello hello")

pyautogui.hotkey("ctrlleft", "shiftleft", "esc")
