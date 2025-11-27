import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import json
import threading
import websocket
import time
import requests
import datetime
import numpy as np
from get_live_stream import WebRTCStreamer

# --- 配置 ---
SERVER_IP = "101.132.172.117"
HTTP_URL = f"http://{SERVER_IP}:8081/api/send-command"
WS_URL = f"ws://{SERVER_IP}:8081/ws/control"
VIDEO_URL = f"http://{SERVER_IP}:8889/live/psdk-client-M350/whep"
CLIENT_ID = "psdk-client-M350"


# ================= 算法接口 =================
def tracking_algorithm_interface(frame):
    h, w, _ = frame.shape
    # 画准星
    cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 0), 2)
    cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 0), 2)
    return frame, None


# ===========================================

class VirtualJoystick(tk.Canvas):
    """虚拟摇杆组件"""

    def __init__(self, master, width=120, height=120, label="Joystick"):
        super().__init__(master, width=width, height=height, bg="#333", highlightthickness=0)
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        self.radius = 25
        self.max_dist = 40

        self.create_oval(self.center_x - self.max_dist, self.center_y - self.max_dist,
                         self.center_x + self.max_dist, self.center_y + self.max_dist, outline="#555", width=2)
        self.stick = self.create_oval(self.center_x - self.radius, self.center_y - self.radius,
                                      self.center_x + self.radius, self.center_y + self.radius, fill="#888", outline="")
        self.create_text(self.center_x, self.height - 10, text=label, fill="white", font=("Arial", 8))

        self.val_x = 0.0
        self.val_y = 0.0

        self.bind("<Button-1>", self.update_stick)
        self.bind("<B1-Motion>", self.update_stick)
        self.bind("<ButtonRelease-1>", self.reset_stick)

    def update_stick(self, event):
        dx = event.x - self.center_x
        dy = event.y - self.center_y
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist > self.max_dist:
            dx = (dx / dist) * self.max_dist
            dy = (dy / dist) * self.max_dist

        self.coords(self.stick,
                    self.center_x + dx - self.radius, self.center_y + dy - self.radius,
                    self.center_x + dx + self.radius, self.center_y + dy + self.radius)

        self.val_x = dx / self.max_dist
        # Canvas Y坐标向下为正，但我们习惯上推为正
        self.val_y = -dy / self.max_dist

    def reset_stick(self, event):
        self.coords(self.stick,
                    self.center_x - self.radius, self.center_y - self.radius,
                    self.center_x + self.radius, self.center_y + self.radius)
        self.val_x = 0.0
        self.val_y = 0.0


class DroneControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Drone Station")
        self.root.geometry("1100x850")
        self.root.configure(bg="#222")

        self.streamer = WebRTCStreamer(VIDEO_URL)
        self.streamer.start()

        self.ws = None
        self.ws_thread = threading.Thread(target=self.connect_ws, daemon=True)
        self.ws_thread.start()

        self.is_executing_auto_command = False

        # --- UI 布局 ---
        # 1. 视频区域
        self.video_frame = tk.Frame(root, bg="black")
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(self.video_frame, bg="black", text="正在连接视频流...")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # 2. 控制面板
        control_panel = tk.Frame(root, bg="#222")
        control_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # 日志
        log_frame = tk.LabelFrame(control_panel, text="日志", bg="#222", fg="white")
        log_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.log_text = tk.Text(log_frame, height=10, width=35, bg="#333", fg="#0f0", font=("Consolas", 8))
        self.log_text.pack()

        # 按钮
        btn_frame = tk.Frame(control_panel, bg="#222")
        btn_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        self.create_btn(btn_frame, "起飞", self.cmd_takeoff, "#28a745", 0, 0)
        self.create_btn(btn_frame, "降落", self.cmd_land, "#dc3545", 0, 1)
        self.create_btn(btn_frame, "确认降落", self.cmd_confirm, "#ffc107", 1, 0, text_color="black")
        self.create_btn(btn_frame, "开启控制权", self.cmd_enable, "#17a2b8", 1, 1)
        self.create_btn(btn_frame, "紧急悬停", self.cmd_hover, "#6c757d", 2, 0)

        # 手动指令 (标签已更新为新定义)
        manual_frame = tk.LabelFrame(control_panel, text="定速指令测试 (50Hz)", bg="#222", fg="white")
        manual_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        self.entries = {}
        # X:前, Y:右, Z:上, Yaw:逆时针
        labels = ["X (前+)", "Y (右+)", "Z (上+)", "Yaw (逆+)", "时间(s)"]
        defaults = ["0.0", "0.0", "0.0", "0.0", "2.0"]
        keys = ["x", "y", "z", "yaw", "time"]

        for i, (lbl, key) in enumerate(zip(labels, keys)):
            tk.Label(manual_frame, text=lbl, bg="#222", fg="#aaa").grid(row=i, column=0, sticky="e", padx=2)
            entry = tk.Entry(manual_frame, width=6)
            entry.insert(0, defaults[i])
            entry.grid(row=i, column=1, padx=2, pady=2)
            self.entries[key] = entry

        tk.Button(manual_frame, text="执行指令", bg="#d63384", fg="white",
                  command=self.start_manual_command_thread).grid(row=5, column=0, columnspan=2, pady=5, sticky="we")

        # 摇杆 (标签已更新)
        joy_frame = tk.Frame(control_panel, bg="#222")
        joy_frame.pack(side=tk.RIGHT, padx=5)
        # 左摇杆: 高度 + 旋转
        self.joy_left = VirtualJoystick(joy_frame, label="左: Z(上) / Yaw(旋)")
        self.joy_left.pack(side=tk.LEFT, padx=10)
        # 右摇杆: 前后 + 左右
        self.joy_right = VirtualJoystick(joy_frame, label="右: X(前) / Y(右)")
        self.joy_right.pack(side=tk.LEFT, padx=10)

        # 循环
        self.update_video_loop()
        self.send_velocity_loop()

    def log(self, msg):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{t}] {msg}\n")
        self.log_text.see(tk.END)

    def create_btn(self, parent, text, cmd, color, row, col, text_color="white"):
        tk.Button(parent, text=text, command=cmd, bg=color, fg=text_color, width=10).grid(row=row, column=col, padx=2,
                                                                                          pady=2)

    def send_vel_command(self, x, y, z, yaw):
        """
        最终修正版发送函数：
        参数定义：
          x: 向前 (Forward+)
          y: 向右 (Right+)
          z: 向上 (Up+)
          yaw: 逆时针旋转 (CCW+)

        后端映射 (基于之前的测试)：
          left_stick_y -> 控制前后
          left_stick_x -> 控制左右
          right_stick_y -> 控制升降
          right_stick_x -> 控制旋转
        """
        # 打印调试: 确认传入的值是否符合预期
        # print(f"CMD -> Forward(x):{x}, Right(y):{y}, Up(z):{z}, Turn(yaw):{yaw}")

        payload = {
            "command": "vstick_advance_v",
            "data": {
                # 1. 前后控制：使用 x 参数
                "left_stick_y": float(x),

                # 2. 左右控制：使用 y 参数
                "left_stick_x": float(y),

                # 3. 升降控制：使用 z 参数
                "right_stick_y": float(z),

                # 4. 旋转控制：使用 yaw 参数 (注意负号，适配逆时针为正)
                "right_stick_x": -float(yaw)
            }
        }
        if self.ws and self.ws.sock and self.ws.sock.connected:
            msg = {"client_id": CLIENT_ID, "payload": payload}
            try:
                self.ws.send(json.dumps(msg))
            except Exception:
                pass

    def start_manual_command_thread(self):
        try:
            x = float(self.entries["x"].get())
            y = float(self.entries["y"].get())
            z = float(self.entries["z"].get())
            yaw = float(self.entries["yaw"].get())
            duration = float(self.entries["time"].get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
            return

        self.log(f"开始执行: X={x}, Y={y}, Z={z}, Yaw={yaw}, T={duration}s")

        def _run():
            self.is_executing_auto_command = True
            start_time = time.time()
            while time.time() - start_time < duration:
                self.send_vel_command(x, y, z, yaw)
                time.sleep(0.02)
            self.send_vel_command(0, 0, 0, 0)
            self.is_executing_auto_command = False
            self.log(f"执行完毕")

        threading.Thread(target=_run, daemon=True).start()

    def send_velocity_loop(self):
        if not self.is_executing_auto_command:
            # === 左摇杆：控制高度(Z) 和 旋转(Yaw) ===
            # 上下推(val_y) -> Z (向上)
            joy_z = self.joy_left.val_y * 2.0

            # 左右推(val_x) -> Yaw (旋转)
            # 向左推(val_x < 0) -> 期望逆时针(Yaw > 0) -> 所以取反
            joy_yaw = -self.joy_left.val_x * 30.0

            # === 右摇杆：控制前进(X) 和 向右(Y) ===
            # 上下推(val_y) -> X (向前)
            joy_x_forward = self.joy_right.val_y * 3.0

            # 左右推(val_x) -> Y (向右)
            joy_y_right = self.joy_right.val_x * 3.0

            # 调用 send_vel_command(x, y, z, yaw)
            self.send_vel_command(joy_x_forward, joy_y_right, joy_z, joy_yaw)

        self.root.after(100, self.send_velocity_loop)

    # --- 视频 & HTTP 部分保持不变 ---
    def update_video_loop(self):
        frame = self.streamer.get_latest_frame()
        if frame is not None:
            frame, _ = tracking_algorithm_interface(frame)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)

            win_w = self.video_frame.winfo_width()
            win_h = self.video_frame.winfo_height()

            if win_w > 10 and win_h > 10:
                img_ratio = img.width / img.height
                win_ratio = win_w / win_h
                if img_ratio > win_ratio:
                    new_w = win_w
                    new_h = int(win_w / img_ratio)
                else:
                    new_h = win_h
                    new_w = int(win_h * img_ratio)
                img = img.resize((new_w, new_h), Image.Resampling.NEAREST)

                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk, text="")

        self.root.after(30, self.update_video_loop)

    def send_http_command(self, command_str):
        def _send():
            try:
                self.log(f"发送HTTP: {command_str}")
                requests.post(HTTP_URL, json={"client_id": CLIENT_ID, "command": command_str}, timeout=2)
            except Exception as e:
                self.log(f"HTTP错误: {e}")

        threading.Thread(target=_send).start()

    def cmd_takeoff(self):
        self.send_http_command("takeoff")

    def cmd_land(self):
        self.send_http_command("land")

    def cmd_confirm(self):
        self.send_http_command("confirm_land")

    def cmd_enable(self):
        self.send_http_command("enable_vstick")

    def cmd_hover(self):
        self.joy_left.reset_stick(None)
        self.joy_right.reset_stick(None)
        self.send_vel_command(0, 0, 0, 0)
        self.log("手动悬停 (重置摇杆)")

    def connect_ws(self):
        self.ws = websocket.WebSocketApp(WS_URL,
                                         on_open=lambda ws: self.log("WebSocket 已连接"),
                                         on_error=lambda ws, e: None)
        self.ws.run_forever()

    def on_close(self):
        self.streamer.stop()
        if self.ws: self.ws.close()
        import os
        os._exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = DroneControlApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()