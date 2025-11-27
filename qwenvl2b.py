import os
import time
import torch
import cv2
import numpy as np
import re
import threading
import queue
import json
from collections import deque
from PIL import Image

# ModelScope 引用
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor


# ==========================================
# 1. 基础工具 & FPS 控制器 (新增)
# ==========================================
class VideoFPSController:
    """
    用于控制视频播放速度，使其接近原始 FPS
    """

    def __init__(self, cap):
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0  # 默认兜底
        self.target_interval = 1.0 / self.fps
        print(f"[System] Video FPS: {self.fps:.2f}, Target Interval: {self.target_interval:.4f}s")

    def sync(self, start_time):
        """
        计算需要等待多久才能保持 1倍速
        """
        process_duration = time.time() - start_time
        wait_time_sec = self.target_interval - process_duration

        # 转换为毫秒，至少等待 1ms
        wait_ms = max(1, int(wait_time_sec * 1000))
        return wait_ms


# ==========================================
# 2. Qwen-VL 封装器 (含 Resize 优化)
# ==========================================
class QwenWrapper:
    def __init__(self, model_path):
        print(f"[System] Loading Qwen Model from: {model_path} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype="auto",
                device_map="auto",
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_path)
            print("[System] Qwen Model Loaded.")
        except Exception as e:
            print(f"[Error] Load failed: {e}")
            raise e

    def _load_image(self, image_input, max_size=1024):
        """
        加载并Resize图片，避免送入过大分辨率导致推理极慢
        """
        if isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.open(image_input).convert("RGB")

        # 优化：按比例缩小图片，最大边不超过 max_size
        w, h = pil_img.size
        if w > max_size or h > max_size:
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return pil_img

    def detect_objects(self, image_input, target_text):
        # 注意：这里加载的是可能 resize 过的图片
        image = self._load_image(image_input)
        w_img, h_img = image.size  # 获取 resize 后的尺寸

        prompt_text = (
            f"Detect '{target_text}'. "
            "Output the result in JSON format only. "
            "The JSON should contain a list of objects with 'label' and 'bbox_2d'. "
            "The 'bbox_2d' must be [xmin, ymin, xmax, ymax] normalized to 1000."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        results = []

        # 获取原始输入的尺寸，用于最终坐标还原
        if isinstance(image_input, np.ndarray):
            orig_h, orig_w = image_input.shape[:2]
        else:
            orig_img = Image.open(image_input)
            orig_w, orig_h = orig_img.size

        try:
            clean_text = output_text.replace("<|im_end|>", "").strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()
            elif "[" in clean_text and "]" in clean_text:
                start = clean_text.find("[")
                end = clean_text.rfind("]") + 1
                clean_text = clean_text[start:end]

            data = json.loads(clean_text)

            for item in data:
                if "bbox_2d" in item:
                    x1_n, y1_n, x2_n, y2_n = item["bbox_2d"]

                    # 坐标转换：
                    # 1. 归一化 -> Resize后的尺寸
                    # 2. 实际上 Qwen 输出的是相对于输入 image 的坐标
                    #    所以我们只需要把 normalized(0-1000) 映射回 原始尺寸(orig_w, orig_h)

                    x1 = x1_n * orig_w / 1000
                    y1 = y1_n * orig_h / 1000
                    x2 = x2_n * orig_w / 1000
                    y2 = y2_n * orig_h / 1000

                    bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    results.append({"label": target_text, "bbox": bbox})

        except Exception as e:
            # print(f"[Warning] JSON parse failed, trying regex... {e}")
            pattern = r"\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)"
            matches = re.findall(pattern, output_text)
            for match in matches:
                val1, val2, val3, val4 = map(int, match)
                x1 = val1 * orig_w / 1000
                y1 = val2 * orig_h / 1000
                x2 = val3 * orig_w / 1000
                y2 = val4 * orig_h / 1000
                bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                results.append({"label": target_text, "bbox": bbox})

        return results

    def ask_question(self, image_input, question):
        t0 = time.perf_counter()
        image = self._load_image(image_input)  # 同样使用 resize 加速 VQA
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0]
        return output_text, (time.perf_counter() - t0)


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
# 4. 异步 Worker (保持不变)
# ==========================================
class AsyncQwenWorker:
    def __init__(self, model):
        self.model = model
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        print("[System] Qwen Background Worker Started.")
        while self.running:
            try:
                task = self.input_queue.get(timeout=0.5)
                frame, prompt, capture_time = task

                # Qwen 推理
                results = self.model.detect_objects(frame, prompt)

                self.output_queue.put((results, capture_time))

                # 保持队列最新，丢弃旧结果
                while self.output_queue.qsize() > 1:
                    self.output_queue.get()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Worker Error] {e}")

    def submit_task(self, frame, prompt):
        if self.input_queue.empty():
            # 传入 capture_time 用于后续对齐轨迹
            self.input_queue.put((frame.copy(), prompt, time.time()))

    def get_result(self):
        if not self.output_queue.empty(): return self.output_queue.get()
        return None

    def stop(self):
        self.running = False
        self.thread.join()


# ==========================================
# 5. 混合追踪器 (优化：自动重找回)
# ==========================================
# ==========================================
# 5. 混合追踪器 (加入平滑滤波抗抖动版)
# ==========================================
class HybridTracker:
    def __init__(self, model, correction_interval=1.0):
        self.worker = AsyncQwenWorker(model)
        self.correction_interval = correction_interval
        self.traj_buffer = TrajectoryBuffer(max_duration=4.0)
        self.tracker = None
        self.is_tracking = False
        self.current_bbox = None  # 格式: (x, y, w, h)
        self.target_prompt = None
        self.last_submit_time = 0

        # 【新增】平滑系数 (0.1 ~ 1.0)，越小越平滑，但响应越慢；越大越灵敏，但越抖动
        self.smooth_factor = 0.3

    def _xyxy_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    def _calculate_iou(self, boxA, boxB):
        # box: (x, y, w, h)
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def _smooth_bbox(self, old_box, new_box, factor):
        """ 计算加权平均值来实现平滑移动 """
        ox, oy, ow, oh = old_box
        nx, ny, nw, nh = new_box

        # 线性插值
        rx = int(ox * (1 - factor) + nx * factor)
        ry = int(oy * (1 - factor) + ny * factor)
        rw = int(ow * (1 - factor) + nw * factor)
        rh = int(oh * (1 - factor) + nh * factor)

        return (rx, ry, rw, rh)

    def init_smart(self, frame, user_input):
        print(f"[Init] Asking Qwen to detect: '{user_input}'...")
        results = self.worker.model.detect_objects(frame, user_input)

        if not results:
            print("[Init] Qwen found nothing.")
            return False, None

        target = results[0]
        bbox_xywh = self._xyxy_to_xywh(target['bbox'])

        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox_xywh)
        self.is_tracking = True
        self.current_bbox = bbox_xywh
        self.target_prompt = user_input
        self.last_submit_time = time.time()
        self.traj_buffer.add(bbox_xywh)
        return True, bbox_xywh

    def update(self, frame):
        # 1. 安全检查：无目标则空转
        if not self.target_prompt:
            return False, None, "Idle", 0

        t_start = time.perf_counter()
        status = "Tracking"

        # 2. CSRT 实时追踪 (高频，作为基础)
        if self.is_tracking:
            ok, bbox = self.tracker.update(frame)
            if ok:
                self.current_bbox = bbox
                self.traj_buffer.add(bbox)
                status = "CSRT"
            else:
                self.is_tracking = False
                status = "Lost"
        else:
            status = "Lost"

        # 3. 异步修正 (低频，作为校准)
        async_res = self.worker.get_result()
        if async_res:
            results, capture_timestamp = async_res
            if results:
                qwen_bbox_raw = results[0]['bbox']
                qwen_bbox_xywh = self._xyxy_to_xywh(qwen_bbox_raw)

                should_hard_reset = False  # 是否强制重置（用于找回丢失目标）
                should_soft_correct = False  # 是否平滑修正（用于微调）

                if not self.is_tracking:
                    # 之前跟丢了，现在找回来了 -> 强制重置
                    should_hard_reset = True
                    status = "Re-found"
                else:
                    # 正在跟，计算偏差
                    iou = self._calculate_iou(self.current_bbox, qwen_bbox_xywh)

                    # 【优化策略】
                    # 情况A: IOU > 0.6 -> CSRT跟得很紧，Qwen只是微小抖动 -> 【忽略Qwen，不修正】
                    # 情况B: 0.2 < IOU < 0.6 -> 有偏差，但还没丢 -> 【平滑修正】
                    # 情况C: IOU < 0.2 -> 偏差极大，可能是CSRT漂移了 -> 【强制重置】

                    if iou > 0.6:
                        # 信任 CSRT，什么都不做，防止不动时闪烁
                        pass
                    elif iou > 0.2:
                        should_soft_correct = True
                        status = "Correcting..."
                    else:
                        should_hard_reset = True
                        status = "Resetting"

                # 计算运动补偿 (因为 Qwen 的结果是 1秒前的)
                dx, dy = self.traj_buffer.get_delta(capture_timestamp,
                                                    self.current_bbox if self.current_bbox else qwen_bbox_xywh)

                # 补偿后的 Qwen 目标位置
                compensated_bbox = (
                    int(qwen_bbox_xywh[0] + dx),
                    int(qwen_bbox_xywh[1] + dy),
                    int(qwen_bbox_xywh[2]),
                    int(qwen_bbox_xywh[3])
                )

                # 执行更新
                if should_hard_reset:
                    # 强制重置：直接跳过去
                    final_bbox = compensated_bbox
                    self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(frame, final_bbox)
                    self.current_bbox = final_bbox
                    self.is_tracking = True

                elif should_soft_correct:
                    # 平滑修正：在当前框和 Qwen 框之间取中间值
                    # 使用 smooth_factor 控制“吸附”力度
                    final_bbox = self._smooth_bbox(self.current_bbox, compensated_bbox, self.smooth_factor)

                    # 修正 CSRT 的内部状态（这一步很重要，否则下一帧 CSRT 又会跳回去）
                    self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(frame, final_bbox)
                    self.current_bbox = final_bbox

        # 4. 提交新任务
        if self.target_prompt and (time.time() - self.last_submit_time > self.correction_interval):
            self.worker.submit_task(frame, self.target_prompt)
            self.last_submit_time = time.time()

        return self.is_tracking, self.current_bbox, status, (time.perf_counter() - t_start)


# ==========================================
# 6. 主程序
# ==========================================
def run_demo():
    # 路径配置
    MODEL_PATH = r"C:\Users\qijiu\.cache\modelscope\hub\models\Qwen\Qwen3-VL-2B-Instruct"
    VIDEO_PATH = "1.mp4"

    if not os.path.exists(VIDEO_PATH):
        print(f"[Error] Video not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[Error] Failed to open video.")
        return

    # 1. 初始化 FPS 控制器
    fps_controller = VideoFPSController(cap)

    # 2. 加载模型
    try:
        qwen_wrapper = QwenWrapper(MODEL_PATH)
    except Exception as e:
        print(f"[Error] Model load failed: {e}")
        return

    # 3. 初始化追踪器
    tracker = HybridTracker(qwen_wrapper, correction_interval=1.0)

    ret, frame = cap.read()
    if not ret: return

    print("\n" + "=" * 40)
    print(" QWEN-VL TRACKER READY ")
    print("=" * 40)

    # 获取初始目标
    user_cmd = input("Target to track (e.g., 'car', 'person'): ").strip()
    if not user_cmd: user_cmd = "car"

    success, _ = tracker.init_smart(frame, user_cmd)
    if not success:
        print("[Error] Init failed. Object not found.")
        tracker.shutdown()
        return

    print("[Info] Tracking started. Press 'q' to quit, 'space' for VQA.")

    try:
        while True:
            # 记录循环开始时间，用于 FPS 同步
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[Info] End of video.")
                break

            display_frame = frame.copy()

            # 追踪更新
            is_tracking, bbox, status, lat = tracker.update(frame)

            if is_tracking and bbox:
                x, y, w, h = [int(v) for v in bbox]
                color = (0, 255, 255) if "Corrected" in status or "Re-found" in status else (0, 255, 0)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                # 状态栏
                info = f"Target: {user_cmd} | {status} | Lat: {lat * 1000:.0f}ms"
                cv2.putText(display_frame, info, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(display_frame, f"Lost: {user_cmd} (Searching...)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

            cv2.imshow("Qwen Tracker", display_frame)

            # 计算需要等待的时间以保持 1倍速
            wait_ms = fps_controller.sync(loop_start)

            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                # 暂停视频进行 VQA
                cv2.putText(display_frame, "Thinking...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("Qwen Tracker", display_frame)
                cv2.waitKey(1)

                q = input("\n[VQA] Question: ")
                if not q: q = "Describe this object."

                # 裁剪出当前追踪的物体图（如果有），否则用全图
                query_img = frame
                if is_tracking and bbox:
                    x, y, w, h = [int(v) for v in bbox]
                    # 稍微外扩一点
                    h_img, w_img = frame.shape[:2]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w_img, x + w), min(h_img, y + h)
                    if x2 > x1 and y2 > y1:
                        query_img = frame[y1:y2, x1:x2]

                ans, t = qwen_wrapper.ask_question(query_img, q)
                print(f"Qwen: {ans} ({t:.2f}s)")
                print("-" * 30)

    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tracker.shutdown()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()