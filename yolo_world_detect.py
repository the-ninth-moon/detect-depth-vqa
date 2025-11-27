import os
import time
import torch
import cv2
import numpy as np
import threading
import queue
from collections import deque
from PIL import Image

# 请先安装: pip install translators
import translators as ts

from ultralytics import YOLO


# ==========================================
# 0. 【内部工具】翻译模块
# ==========================================
class Translator:
    """
    一个简单的翻译器，将中文文本翻译成英文。
    它将被 HybridTracker 内部使用。
    """

    def __init__(self):
        # 这个初始化信息只在启动时显示一次，告知系统具备该能力
        print("[System] Translator Initialized (Engine: Google).")

    def translate(self, text, target_lang='en'):
        if not text:
            return ""
        try:
            # 内部进行翻译
            translated_text = ts.translate_text(query_text=text, to_language=target_lang)
            return translated_text
        except Exception as e:
            print(f"[Warning] Translation failed for '{text}': {e}. Falling back to original text.")
            return text


# ==========================================
# 1. 基础工具 & FPS 控制器 (保持不变)
# ==========================================
class VideoFPSController:
    def __init__(self, cap):
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0
        self.target_interval = 1.0 / self.fps
        print(f"[System] Video FPS: {self.fps:.2f}, Target Interval: {self.target_interval:.4f}s")

    def sync(self, start_time):
        process_duration = time.time() - start_time
        wait_time_sec = self.target_interval - process_duration
        wait_ms = max(1, int(wait_time_sec * 1000))
        return wait_ms


# ==========================================
# 2. YOLO-World 封装器 (保持不变)
# ==========================================
class YOLOWorldWrapper:
    def __init__(self, model_path, conf_threshold=0.1):
        print(f"[System] Loading YOLO-World Model from: {model_path} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        try:
            self.model = YOLO(model_path)
            print(f"[System] YOLO-World Model Loaded on {self.device}.")
        except Exception as e:
            print(f"[Error] YOLO-World load failed: {e}")
            raise e

    def detect_objects(self, image_input, target_text):
        self.model.set_classes([target_text])
        results = self.model.predict(image_input, conf=self.conf_threshold, device=self.device, verbose=False)
        formatted_results = []
        boxes = results[0].boxes.cpu().numpy()
        for box in boxes:
            bbox_coords = box.xyxy[0].tolist()
            formatted_results.append({
                "label": target_text,
                "bbox": bbox_coords
            })
        if formatted_results:
            return [formatted_results[0]]
        else:
            return []


# ==========================================
# 3. 轨迹缓冲区 (保持不变)
# ==========================================
class TrajectoryBuffer:
    def __init__(self, max_duration=2.0):
        self.buffer = deque()
        self.max_duration = max_duration

    def add(self, bbox):
        now = time.time()
        self.buffer.append((now, bbox))
        while self.buffer and now - self.buffer[0][0] > self.max_duration:
            self.buffer.popleft()

    def get_delta(self, target_time, current_bbox):
        if not self.buffer: return 0, 0
        closest_bbox = None
        min_diff = float('inf')
        for t, bbox in self.buffer:
            diff = abs(t - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_bbox = bbox
        if min_diff > 1.0 or closest_bbox is None: return 0, 0
        cur_x, cur_y, _, _ = current_bbox
        old_x, old_y, _, _ = closest_bbox
        return (cur_x - old_x), (cur_y - old_y)


# ==========================================
# 4. 异步 Worker (逻辑不变)
# ==========================================
class AsyncDetectorWorker:
    def __init__(self, model):
        self.model = model
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        print("[System] Detector Background Worker Started.")
        while self.running:
            try:
                task = self.input_queue.get(timeout=0.5)
                frame, prompt, capture_time = task
                results = self.model.detect_objects(frame, prompt)
                if self.output_queue.full():
                    self.output_queue.get_nowait()
                self.output_queue.put((results, capture_time))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Worker Error] {e}")

    def submit_task(self, frame, prompt):
        try:
            self.input_queue.put_nowait((frame.copy(), prompt, time.time()))
        except queue.Full:
            pass

    def get_result(self):
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        self.thread.join()
        print("[System] Detector Worker Stopped.")


# ==========================================
# 5. 【核心修改】混合追踪器 (封装翻译功能)
# ==========================================
class HybridTracker:
    def __init__(self, model, correction_interval=0.3):
        # 内部实例化并持有 Translator
        self.translator = Translator()
        self.worker = AsyncDetectorWorker(model)
        self.correction_interval = correction_interval
        self.traj_buffer = TrajectoryBuffer(max_duration=4.0)
        self.tracker = None
        self.is_tracking = False
        self.current_bbox = None

        self.original_prompt = None  # 保存用户输入的原始文本 (例如中文)
        self.target_prompt = None  # 保存翻译后的英文文本，用于模型推理

        self.last_submit_time = 0
        self.smooth_factor = 0.3

    def _xyxy_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]);
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3];
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def _smooth_bbox(self, old_box, new_box, factor):
        ox, oy, ow, oh = old_box;
        nx, ny, nw, nh = new_box
        rx = int(ox * (1 - factor) + nx * factor);
        ry = int(oy * (1 - factor) + ny * factor)
        rw = int(ow * (1 - factor) + nw * factor);
        rh = int(oh * (1 - factor) + nh * factor)
        return (rx, ry, rw, rh)

    def init_smart(self, frame, user_prompt):
        """
        初始化追踪器，接口直接接收用户输入（可以是中文）。
        翻译过程在此方法内部完成，对调用者透明。
        """
        # 1. 保存原始输入并进行翻译
        self.original_prompt = user_prompt
        print(f"[Tracker-Internal] Received prompt: '{user_prompt}'. Translating...")
        self.target_prompt = self.translator.translate(user_prompt)
        print(f"[Tracker-Internal] Translated to: '{self.target_prompt}'.")

        if not self.target_prompt:
            print("[Error] Translation result is empty. Cannot initialize tracker.")
            return False, None

        # 2. 使用翻译后的英文 prompt 进行目标检测
        print(f"[Init] Asking Detector to find: '{self.target_prompt}'...")
        results = self.worker.model.detect_objects(frame, self.target_prompt)

        if not results:
            print("[Init] Detector found nothing.")
            return False, None

        # 3. 初始化追踪
        target = results[0]
        bbox_xywh = self._xyxy_to_xywh(target['bbox'])
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox_xywh)
        self.is_tracking = True
        self.current_bbox = bbox_xywh
        self.last_submit_time = time.time()
        self.traj_buffer.add(bbox_xywh)
        return True, bbox_xywh

    def update(self, frame):
        if not self.target_prompt:
            return False, None, "Idle", 0

        t_start = time.perf_counter()
        status = "Tracking"

        if self.is_tracking:
            ok, bbox = self.tracker.update(frame)
            if ok:
                self.current_bbox = bbox
                self.traj_buffer.add(bbox)
                status = "CSRT"
            else:
                self.is_tracking = False;
                status = "Lost"
        else:
            status = "Lost"

        async_res = self.worker.get_result()
        if async_res:
            results, capture_timestamp = async_res
            if results:
                detector_bbox_xywh = self._xyxy_to_xywh(results[0]['bbox'])
                should_hard_reset = not self.is_tracking
                status = "Re-found" if should_hard_reset else status

                if self.is_tracking:
                    iou = self._calculate_iou(self.current_bbox, detector_bbox_xywh)
                    if iou > 0.2:
                        dx, dy = self.traj_buffer.get_delta(capture_timestamp, self.current_bbox)
                        compensated_bbox = (
                        int(detector_bbox_xywh[0] + dx), int(detector_bbox_xywh[1] + dy), detector_bbox_xywh[2],
                        detector_bbox_xywh[3])
                        final_bbox = self._smooth_bbox(self.current_bbox, compensated_bbox, self.smooth_factor)
                        self.tracker = cv2.TrackerCSRT_create();
                        self.tracker.init(frame, final_bbox)
                        self.current_bbox = final_bbox;
                        status = "Correcting"
                    else:
                        should_hard_reset = True;
                        status = "Resetting"

                if should_hard_reset:
                    dx, dy = self.traj_buffer.get_delta(capture_timestamp,
                                                        self.current_bbox if self.current_bbox else detector_bbox_xywh)
                    final_bbox = (
                    int(detector_bbox_xywh[0] + dx), int(detector_bbox_xywh[1] + dy), detector_bbox_xywh[2],
                    detector_bbox_xywh[3])
                    self.tracker = cv2.TrackerCSRT_create();
                    self.tracker.init(frame, final_bbox)
                    self.current_bbox = final_bbox;
                    self.is_tracking = True

        if self.target_prompt and (time.time() - self.last_submit_time > self.correction_interval):
            self.worker.submit_task(frame, self.target_prompt)
            self.last_submit_time = time.time()

        return self.is_tracking, self.current_bbox, status, (time.perf_counter() - t_start)

    def shutdown(self):
        print("[System] Shutting down tracker...")
        self.worker.stop()


# ==========================================
# 6. 主程序 (接口调用更简洁)
# ==========================================
def run_demo():
    MODEL_PATH = 'yolov8s-world.pt'
    VIDEO_PATH = "1.mp4"

    if not os.path.exists(VIDEO_PATH):
        print(f"[Error] Video not found: {VIDEO_PATH}");
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[Error] Failed to open video.");
        return

    fps_controller = VideoFPSController(cap)

    try:
        # 只需要初始化模型和追踪器
        detector_wrapper = YOLOWorldWrapper(MODEL_PATH)
        tracker = HybridTracker(detector_wrapper, correction_interval=0.3)
    except Exception as e:
        print(f"[Error] Initialization failed: {e}");
        return

    ret, frame = cap.read()
    if not ret: return

    print("\n" + "=" * 40)
    print(" YOLO-WORLD 混合追踪器 (支持中文输入)")
    print("=" * 40)

    # 主程序逻辑非常干净：直接获取用户输入
    user_prompt = input("请输入要追踪的目标 (例如: '一辆车', '穿红衬衫的人'): ").strip()
    if not user_prompt:
        user_prompt = "一辆车"  # 默认中文目标

    # 直接将用户输入传给 tracker，不再关心翻译细节
    success, _ = tracker.init_smart(frame, user_prompt)
    if not success:
        print(f"[Error] 初始化失败. 未在画面中找到 '{user_prompt}'.")
        tracker.shutdown();
        return

    print("[Info] 追踪已开始. 按 'q' 键退出.")

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[Info] 视频播放结束.");
                break

            display_frame = frame.copy()
            is_tracking, bbox, status, lat = tracker.update(frame)

            if is_tracking and bbox:
                x, y, w, h = [int(v) for v in bbox]
                color = (0, 255, 255) if "Correcting" in status or "Re-found" in status or "Resetting" in status else (
                0, 255, 0)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                # 从 tracker 获取原始 prompt 来显示，确保显示的是用户的输入
                info = f"目标: {tracker.original_prompt} | {status} | Lat: {lat * 1000:.0f}ms"
                cv2.putText(display_frame, info, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(display_frame, f"丢失目标: {tracker.original_prompt} (搜索中...)", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("YOLO-World Tracker", display_frame)
            wait_ms = fps_controller.sync(loop_start)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'): break

    except KeyboardInterrupt:
        print("用户中断程序。")
    except Exception as e:
        print(f"发生未知错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tracker.shutdown()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()