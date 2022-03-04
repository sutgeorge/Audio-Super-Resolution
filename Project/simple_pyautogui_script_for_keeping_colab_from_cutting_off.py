import pyautogui as pyag
import time

while True:
    pyag.moveTo(600, 400, duration=1)
    pyag.moveTo(1200, 400, duration=1)
    pyag.click(1200, 400, duration=1)
    time.sleep(60)
