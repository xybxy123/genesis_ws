# keyboard_input.py
import sys
import threading
from typing import Optional


class KeyboardInput:
    def __init__(self):
        self.key = None
        self.running = True
        self._thread = None

    def start(self):
        # 启动后台线程监听键盘
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        # Linux / Mac 无回车读取键盘
        try:
            import tty
            import termios

            old_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())

            while self.running:
                self.key = sys.stdin.read(1)
        except:
            # Windows
            import msvcrt

            while self.running:
                if msvcrt.kbhit():
                    self.key = msvcrt.getch().decode()

    def get_key(self) -> Optional[str]:
        # 立刻返回当前按键，不阻塞！
        key = self.key
        self.key = None
        return key

    def stop(self):
        self.running = False
