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
# 1. 引入所有AI模块
# =========================================================

try:
    from yolo_world_detect import YOLOWorldWrapper, HybridTracker as YoloHybridTracker
    print("[System] Imported YOLO-World tracker.")
    from qwenvl2b import QwenWrapper
    print("[System] Imported Qwen-VL VQA module.")
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


# VideoLabel (保持不变)
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
# 2. VQA 工作线程 (使用 Qwen-VL)
# =========================================================

class VQAWorker(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, qwen_wrapper):
        super().__init__()
        self.qwen_wrapper = qwen_wrapper
        self.image = None
        self.question = ""

    def run(self):
        if self.image is None or not self.question: return
        try:
            ans, t = self.qwen_wrapper.ask_question(self.image, self.question)
            self.result_ready.emit(f"<b>Qwen:</b> {ans} <span style='color:gray'>({t:.2f}s)</span>")
        except Exception as e:
            self.result_ready.emit(f"<span style='color:red'>VQA Error: {e}</span>")


# =========================================================
# 3. 视频与追踪线程 (加载三个模型)
# =========================================================

class VideoThread(QThread):
    # 【改动】只保留主视频帧信号
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

    def run(self):
        try:
            self.status_signal.emit("Loading YOLO-World Model...")
            self.detector_wrapper = YOLOWorldWrapper(self.yolo_model_path)
            self.tracker = YoloHybridTracker(self.detector_wrapper, correction_interval=1.0)

            self.status_signal.emit("Loading Qwen-VL Model...")
            self.qwen_wrapper = QwenWrapper(self.qwen_model_path)

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
            # 处理指令
            try:
                cmd_type, cmd_data = self.command_queue.get_nowait()
                if cmd_type == "detect":
                    if self.current_frame is not None:
                        self.status_signal.emit(f"Detecting: {cmd_data}")
                        success, _ = self.tracker.init_smart(self.current_frame, cmd_data)
                        if not success: self.status_signal.emit(f"Failed to find: {cmd_data}")
                elif cmd_type == "manual":
                    if self.current_frame is not None:
                        bbox = cmd_data
                        self.tracker.tracker = cv2.TrackerCSRT_create()
                        self.tracker.tracker.init(self.current_frame, bbox)
                        self.tracker.current_bbox = bbox
                        self.tracker.is_tracking = True
                        self.tracker.original_prompt = "Manual Selection"
                        self.tracker.target_prompt = "object"
                        self.tracker.traj_buffer.buffer.clear()
                        self.tracker.traj_buffer.add(bbox)
                        self.status_signal.emit("Manual Tracking Started")
            except queue.Empty:
                pass

            frame = self.streamer.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            self.current_frame = frame.copy()

            # 1. YOLO 追踪更新
            is_tracking, bbox, status, lat = self.tracker.update(frame)

            # 2. 深度估计与距离计算 (后台依然进行)
            object_distance = None
            if self.depth_estimator:
                current_bbox_for_depth = bbox if is_tracking else None
                # estimate_bbox_depth现在返回距离和可视化的深度图
                # 我们只需要距离，所以用 _ 忽略第二个返回值
                object_distance, _ = self.depth_estimator.estimate_bbox_depth(
                    self.current_frame, current_bbox_for_depth
                )
                print(object_distance)

            # 3. 绘制结果到主画面
            display_frame = self.current_frame.copy()
            if is_tracking and bbox:
                x, y, w, h = [int(v) for v in bbox]
                color = (0, 255, 255) if "Correcting" in status or "Re-found" in status else (0, 255, 0)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                # 准备距离文本，如果计算失败则显示 N/A
                dist_text = "Dist: N/A"
                if object_distance is not None:
                    dist_text = f"Dist: {object_distance:.2f}m"

                # 将所有信息组合在一起显示
                info_text = f"{self.tracker.original_prompt} | {status} | {dist_text}"

                display_frame = cv2_draw_chinese(display_frame, info_text, (x, max(0, y - 30)), color, size=24)

            # 4. 【改动】只发送主视频帧到UI
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)

    def stop(self):
        self.running = False
        if self.streamer: self.streamer.stop()
        if self.tracker: self.tracker.shutdown()
        self.quit()
        self.wait()


# =========================================================
# 4. 主窗口 GUI
# =========================================================

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO-Tracker + Qwen-VQA + Depth-Estimation")
        # 【改动】恢复适合单个视频流的窗口尺寸
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
        self.QWEN_MODEL_PATH = r"C:\Users\qijiu\.cache\modelscope\hub\models\Qwen\Qwen3-VL-2B-Instruct"
        self.DEPTH_MODEL_TYPE = "depth-anything/da3metric-large"
        self.init_ui()

        self.video_thread = VideoThread(self.STREAM_URL, self.YOLO_MODEL_PATH, self.QWEN_MODEL_PATH,
                                        self.DEPTH_MODEL_TYPE)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        # 【改动】移除深度图信号的连接
        self.video_thread.status_signal.connect(self.update_status)
        self.video_thread.start()

        self.vqa_worker = None

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 【改动】恢复为只显示主视频的简单布局
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

        vqa_group = QGroupBox("VQA(Qwen-VL)")
        vqa_layout = QVBoxLayout()
        self.chat_history = QTextBrowser()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("问场景中的内容")
        self.question_input.returnPressed.connect(self.ask_vqa)
        self.btn_ask = QPushButton("问Qwen")
        self.btn_ask.clicked.connect(self.ask_vqa)
        vqa_layout.addWidget(self.chat_history);
        vqa_layout.addWidget(self.question_input)
        vqa_layout.addWidget(self.btn_ask)
        vqa_group.setLayout(vqa_layout)

        control_layout.addWidget(track_group)
        control_layout.addWidget(vqa_group)
        control_layout.addStretch()

        main_layout.addWidget(video_group, stretch=1)
        main_layout.addWidget(control_panel)

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        self.video_label.set_curr_frame(qt_img)

    # 【改动】移除 update_depth_image 槽函数

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
        self.chat_history.append(f"<i>System: Searching for '{target}'...</i>")

    @pyqtSlot(int, int, int, int)
    def handle_manual_selection(self, x, y, w, h):
        bbox = (x, y, w, h)
        self.video_thread.command_queue.put(("manual", bbox))
        self.chat_history.append(f"<i>System: Manual tracking initiated.</i>")

    def ask_vqa(self):
        question = self.question_input.text().strip()
        if not question or not self.video_thread or not self.video_thread.qwen_wrapper:
            self.chat_history.append("<i>System: VQA module not ready or no question.</i>")
            return

        if self.video_thread.current_frame is None:
            self.chat_history.append("<i>System: No video signal.</i>")
            return

        self.chat_history.append(f"<b>You:</b> {question}")
        self.question_input.clear()
        self.btn_ask.setEnabled(False)

        if self.vqa_worker is None or not self.vqa_worker.isRunning():
            self.vqa_worker = VQAWorker(self.video_thread.qwen_wrapper)
            self.vqa_worker.result_ready.connect(self.on_vqa_result)

        current_img = self.video_thread.current_frame.copy()
        tracker = self.video_thread.tracker
        if tracker and tracker.is_tracking and tracker.current_bbox:
            x, y, w, h = [int(v) for v in tracker.current_bbox]
            h_img, w_img = current_img.shape[:2]
            pad = 20
            x1, y1 = max(0, x - pad), max(0, y - pad);
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            if x2 > x1 and y2 > y1:
                current_img = current_img[y1:y2, x1:x2]
                self.chat_history.append("<i>(Focused on tracked object)</i>")

        self.vqa_worker.image = current_img
        self.vqa_worker.question = question
        self.vqa_worker.start()

    @pyqtSlot(str)
    def on_vqa_result(self, result):
        self.chat_history.append(result)
        self.chat_history.moveCursor(self.chat_history.textCursor().End)
        self.btn_ask.setEnabled(True)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())