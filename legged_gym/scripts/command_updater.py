import threading
import json
import math
import time

def get_command_input(prompt, default_value):
    """Get command input from user and return as a single float value."""
    user_input = input(prompt)
    if user_input.strip() == "":
        return default_value
    return float(user_input)

def update_command_range():
    """Continuously prompt the user for height command (fixed or sine range)."""
    default_height = 0.8   # 默认固定高度
    default_range = [0.3, 0.8]  # 默认范围
    default_freq = 0.5     # 默认频率 Hz

    while True:
        try:
            mode = input("选择模式: (f)ixed or (s)ine [default=f]: ").strip().lower()
            if mode == "s":
                h_min = get_command_input("输入最小高度 (m, 默认0.3): ", default_range[0])
                h_max = get_command_input("输入最大高度 (m, 默认0.8): ", default_range[1])
                freq = get_command_input("输入频率 (Hz, 默认0.5): ", default_freq)
                command_interface = {
                    "mode": "sine",
                    "h_min": h_min,
                    "h_max": h_max,
                    "freq": freq,
                    "t0": time.time()
                }
            else:
                height = get_command_input("输入目标高度 (m, 默认0.8): ", default_height)
                command_interface = {
                    "mode": "fixed",
                    "height": height
                }

            with open('command_interface.json', 'w') as file:
                json.dump(command_interface, file)

            print(f"Updated command: {command_interface}")
        except ValueError as e:
            print("Invalid input. Please enter valid numerical values.")
            print(e)

if __name__ == "__main__":
    thread = threading.Thread(target=update_command_range)
    thread.start()