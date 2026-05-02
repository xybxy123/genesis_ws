# keyboard_test.py
from keyboard_input import KeyboardInput
import time

kb = KeyboardInput()
kb.start()

print("==== 实时键盘测试（不用回车！）====")
print("W S A D 控制，Q 退出\n")

while True:
    key = kb.get_key()  # 非阻塞

    if key == "w":
        print("前进 W")
    elif key == "s":
        print("后退 S")
    elif key == "a":
        print("左移 A")
    elif key == "d":
        print("右移 D")
    elif key == "q":
        print("退出程序")
        break

    time.sleep(0.02)  # 控制循环频率

kb.stop()
