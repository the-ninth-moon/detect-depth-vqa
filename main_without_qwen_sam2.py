import sys
import cv2
import time
import numpy as np
import threading
import queue
import asyncio
import logging
import aiohttp

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QTextBrowser, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QRect
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap

# =========================================================
# 1. 引入AI模块 (已移除 Qwen 以节省性能)
# =========================================================

try:
    from yolo_world_detect import YOLOWorldWrapper, FastSAM2Tracker as YoloHybridTracker

    print("[System] Imported YOLO-World + FastSAM2 tracker.")

    # 【改动1】暂时注释掉 Qwen 模块，节省显存
    # from qwenvl2b import QwenWrapper
    # print("[System] Imported Qwen-VL VQA module.")

    from monocular_depth import MonocularDepth

    print("[System] Imported Monocular Depth module.")
except ImportError as e:
    print(f"[Error] Failed to import modules: {e}")
    sys.exit(1)

# WebRTC 和 中文绘制工具 (保持不变)

from aiortc import RTCPeerConnection, RTCSessionDescription
from PIL import Image, ImageDraw, ImageFont

logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("aiohttp").setLevel(logging.ERROR)


def cv2_draw_chinese(img, text, pos, color=(0, 255, 0), size=20):
    if (isinstance(img, np.ndarray)):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("msyh.ttc", size)
        except:
            try:
                font = ImageFont.truetype("simhei.ttf", size)
            except:
                font = ImageFont.load_default()
        rgb_color = (color[2], color[1], color[0])
        draw.text(pos, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class WebRTCStreamer:
    def __init__(self, url):
        self.url, self.frame_queue = url, queue.Queue(maxsize=2)
        self.running, self.thread, self.pc = False, None, None

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1)

    def get_latest_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _run_event_loop(self):
        if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.pc = RTCPeerConnection()

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video": asyncio.ensure_future(self._consume_track(track))

        try:
            loop.run_until_complete(self._connect_whep())
            while self.running: loop.run_until_complete(asyncio.sleep(0.1))
        finally:
            if self.pc and self.pc.connectionState != 'closed': loop.run_until_complete(self.pc.close())
            loop.close()

    async def _connect_whep(self):
        self.pc.addTransceiver("video", direction="recvonly")
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, data=self.pc.localDescription.sdp,
                                    headers={"Content-Type": "application/sdp"}) as resp:
                if resp.status not in [200, 201]: return
                answer = RTCSessionDescription(sdp=await resp.text(), type="answer")
                await self.pc.setRemoteDescription(answer)

    async def _consume_track(self, track):
        while self.running:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                if self.frame_queue.full(): self.frame_queue.get_nowait()
                self.frame_queue.put(img)
            except Exception:
                pass


class VideoLabel(QLabel):
    selection_finished = pyqtSignal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background-color: #000; border: 1px solid #444;")
        self.current_image = self.start_point = self.end_point = None
        self.is_drawing = False

    def set_curr_frame(self, image):
        self.current_image = image;
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.start_point = self.end_point = event.pos(); self.is_drawing = True

    def mouseMoveEvent(self, event):
        if self.is_drawing: self.end_point = event.pos(); self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.end_point = event.pos()
            if self.current_image:
                rect, scale = self._get_display_rect_and_scale()
                mx1, my1 = min(self.start_point.x(), self.end_point.x()), min(self.start_point.y(),
                                                                              self.end_point.y())
                mx2, my2 = max(self.start_point.x(), self.end_point.x()), max(self.start_point.y(),
                                                                              self.end_point.y())
                ix1, iy1 = mx1 - rect.x(), my1 - rect.y()
                ix2, iy2 = mx2 - rect.x(), my2 - rect.y()
                fx, fy = int(ix1 / scale), int(iy1 / scale)
                fw, fh = int((ix2 - ix1) / scale), int((iy2 - iy1) / scale)
                ow, oh = self.current_image.width(), self.current_image.height()
                fx, fy = max(0, fx), max(0, fy)
                fw, fh = min(fw, ow - fx), min(fh, oh - fy)
                if fw > 10 and fh > 10: self.selection_finished.emit(fx, fy, fw, fh)
            self.start_point = self.end_point = None
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.current_image: rect, _ = self._get_display_rect_and_scale(); painter.drawImage(rect, self.current_image)
        if self.is_drawing and self.start_point and self.end_point:
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect)

    def _get_display_rect_and_scale(self):
        if not self.current_image: return QRect(), 1.0
        wl, hl = self.width(), self.height()
        wi, hi = self.current_image.width(), self.current_image.height()
        scale = min(wl / wi, hl / hi) if wi > 0 and hi > 0 else 1.0
        nw, nh = int(wi * scale), int(hi * scale)
        ox, oy = (wl - nw) // 2, (hl - nh) // 2
        return QRect(ox, oy, nw, nh), scale


# =========================================================
# 2. VQA 工作线程 (被禁用)
# =========================================================
# [改动2] VQAWorker 暂时不需要了，或者保留类但不实例化
class VQAWorker(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, qwen_wrapper): super().__init__(); self.qwen_wrapper = qwen_wrapper

    def run(self): pass


# =========================================================
# 3. 视频与追踪线程 (已优化深度估计频率)
# =========================================================

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str)

    def __init__(self, streamer_url, yolo_path, qwen_path, depth_model_type):
        super().__init__()
        self.streamer_url = streamer_url
        self.yolo_model_path = yolo_path
        self.qwen_model_path = qwen_path
        self.depth_model_type = depth_model_type

        self.streamer = self.tracker = self.detector_wrapper = self.qwen_wrapper = None
        self.depth_estimator = None

        self.running = True
        self.current_frame = None
        self.command_queue = queue.Queue()

        # [改动3] 深度估计频率控制
        self.frame_counter = 0
        self.last_depth_value = None

    def run(self):
        try:
            self.status_signal.emit("Loading YOLO-World Model...")
            self.detector_wrapper = YOLOWorldWrapper(self.yolo_model_path)
            self.tracker = YoloHybridTracker(self.detector_wrapper)

            # [改动4] 跳过 Qwen 加载
            # self.status_signal.emit("Loading Qwen-VL Model...")
            # self.qwen_wrapper = QwenWrapper(self.qwen_model_path)
            self.status_signal.emit("Qwen-VL Disabled for Performance.")

            self.status_signal.emit("Loading Depth Model...")
            self.depth_estimator = MonocularDepth(self.depth_model_type)

            self.status_signal.emit("All models loaded.")
        except Exception as e:
            self.status_signal.emit(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return

        self.status_signal.emit("Connecting WebRTC...")
        self.streamer = WebRTCStreamer(self.streamer_url)
        self.streamer.start()
        self.status_signal.emit("Ready.")

        while self.running:
            self.frame_counter += 1

            # 处理指令
            try:
                cmd_type, cmd_data = self.command_queue.get_nowait()
                if cmd_type == "detect":
                    if self.current_frame is not None:
                        self.status_signal.emit(f"Detecting: {cmd_data}")
                        success = self.tracker.init_tracker(self.current_frame, cmd_data)
                        if not success: self.status_signal.emit(f"Failed to find: {cmd_data}")

                elif cmd_type == "manual":
                    if self.current_frame is not None:
                        try:
                            # 直接调用我们在 yolo_world_detect.py 里新写的封装好的方法
                            success = self.tracker.init_with_box(self.current_frame, cmd_data)

                            if success:
                                self.status_signal.emit("Manual Tracking Started (Refined)")
                            else:
                                self.status_signal.emit("Manual Init Failed")

                        except Exception as e:
                            print(f"Manual Init Error: {e}")
                            self.status_signal.emit("Manual Init Error")

            except queue.Empty:
                pass

            frame = self.streamer.get_latest_frame()
            if frame is None:
                time.sleep(0.005)  # 稍微减少sleep
                continue

            self.current_frame = frame.copy()

            # 1. 追踪更新 (每帧都跑，这是实时的关键)
            is_tracking, box_xyxy, mask, status_info = self.tracker.update(frame)

            bbox_xywh = None
            if is_tracking and box_xyxy is not None:
                x1, y1, x2, y2 = box_xyxy
                bbox_xywh = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # 2. 深度估计与距离计算 (【核心优化】降频处理)
            # 只有当正在追踪，且每隔 15 帧 (约0.5秒) 才跑一次深度模型
            # 否则沿用上一次的距离值
            if self.depth_estimator:
                if is_tracking and self.frame_counter % 15 == 0:
                    current_bbox_for_depth = bbox_xywh
                    dist, _ = self.depth_estimator.estimate_bbox_depth(
                        self.current_frame, current_bbox_for_depth
                    )
                    if dist is not None:
                        self.last_depth_value = dist
                elif not is_tracking:
                    self.last_depth_value = None  # 没追踪时清空

            # 3. 绘制结果
            display_frame = self.current_frame.copy()
            if is_tracking and bbox_xywh:
                x, y, w, h = bbox_xywh
                color = (0, 255, 0)

                if mask is not None:
                    full_mask = cv2.resize(mask.astype(np.uint8), (display_frame.shape[1], display_frame.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                    colored_mask = np.zeros_like(display_frame)
                    colored_mask[full_mask > 0] = (0, 100, 255)
                    display_frame = cv2.addWeighted(display_frame, 0.8, colored_mask, 0.2, 0)

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                dist_text = "Dist: Calculating..."
                if self.last_depth_value is not None:
                    dist_text = f"Dist: {self.last_depth_value:.2f}m"

                info_text = f"{self.tracker.target_prompt} | {status_info} | {dist_text}"
                display_frame = cv2_draw_chinese(display_frame, info_text, (x, max(0, y - 30)), color, size=24)

            # 4. 发送图像
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)

    def stop(self):
        self.running = False
        if self.streamer: self.streamer.stop()
        if self.tracker: self.tracker.stop()
        self.quit()
        self.wait()


# =========================================================
# 4. 主窗口 GUI (禁用VQA相关按钮)
# =========================================================

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO-Tracker + FastSAM2 + Depth")
        self.resize(1280, 720)
        self.setStyleSheet("""
            QMainWindow { background-color: #2c3e50; }
            QGroupBox { color: #ecf0f1; font-weight: bold; border: 1px solid #34495e; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            QLabel { color: #ecf0f1; }
            QLineEdit { border: 1px solid #34495e; padding: 5px; background-color: #34495e; color: #ecf0f1; }
            QPushButton { background-color: #3498db; color: white; border: none; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #566573; }
            QTextBrowser { background-color: #34495e; color: #ecf0f1; border: 1px solid #2c3e50; }
        """)

        self.STREAM_URL = "http://101.132.172.117:8889/live/psdk-client-M350/whep"
        self.YOLO_MODEL_PATH = 'yolov8s-world.pt'
        self.QWEN_MODEL_PATH = ""  # 留空
        self.DEPTH_MODEL_TYPE = "depth-anything/da3metric-large"
        self.init_ui()

        self.video_thread = VideoThread(self.STREAM_URL, self.YOLO_MODEL_PATH, self.QWEN_MODEL_PATH,
                                        self.DEPTH_MODEL_TYPE)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.status_signal.connect(self.update_status)
        self.video_thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        video_group = QGroupBox("Live Stream (Draw box to track manually)")
        video_layout = QVBoxLayout()
        self.video_label = VideoLabel()
        self.video_label.selection_finished.connect(self.handle_manual_selection)
        video_layout.addWidget(self.video_label)
        video_group.setLayout(video_layout)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(350)

        track_group = QGroupBox("跟踪与发现")
        track_layout = QVBoxLayout()
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText("输入目标 (e.g., '一辆车', '一个人')")
        self.target_input.returnPressed.connect(self.start_auto_track)
        self.btn_track = QPushButton("搜索并跟踪")
        self.btn_track.clicked.connect(self.start_auto_track)
        self.lbl_status = QLabel("状态: 初始化...")
        track_layout.addWidget(QLabel("目标物体:"))
        track_layout.addWidget(self.target_input)
        track_layout.addWidget(self.btn_track)
        track_layout.addWidget(self.lbl_status)
        track_group.setLayout(track_layout)

        # [改动5] 禁用 VQA UI 或提示已禁用
        vqa_group = QGroupBox("VQA (Disabled for Performance)")
        vqa_layout = QVBoxLayout()
        self.chat_history = QTextBrowser()
        self.chat_history.append("<i>System: Qwen-VL is disabled to save VRAM.</i>")
        vqa_layout.addWidget(self.chat_history);
        vqa_group.setLayout(vqa_layout)

        control_layout.addWidget(track_group)
        control_layout.addWidget(vqa_group)
        control_layout.addStretch()

        main_layout.addWidget(video_group, stretch=1)
        main_layout.addWidget(control_panel)

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        self.video_label.set_curr_frame(qt_img)

    @pyqtSlot(str)
    def update_status(self, text):
        self.lbl_status.setText(f"Status: {text}")
        if "Error" in text or "Failed" in text:
            self.lbl_status.setStyleSheet("color: #ff5555;")
        else:
            self.lbl_status.setStyleSheet("color: #55ff55;")

    def start_auto_track(self):
        target = self.target_input.text().strip()
        if not target: return
        self.video_thread.command_queue.put(("detect", target))

    @pyqtSlot(int, int, int, int)
    def handle_manual_selection(self, x, y, w, h):
        bbox = (x, y, w, h)
        self.video_thread.command_queue.put(("manual", bbox))

    def ask_vqa(self):
        pass  # Disabled

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())