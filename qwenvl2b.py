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

# ==========================================
# 恢复你的原始引用，不做修改
# ==========================================
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor


# ==========================================
# 0. 【新增】中文指令解析工具 (纯逻辑，不影响模型)
# ==========================================
class CommandParser:
    """
    解析中文指令，提取目标名称和索引逻辑。
    """
    CN_NUM = {'一': 0, '二': 1, '两': 1, '三': 2, '四': 3, '五': 4, '六': 5, '七': 6, '八': 7, '九': 8, '十': 9}

    @staticmethod
    def parse(text):
        text = text.strip()
        index = 0
        reverse = False  # 是否倒序

        # 1. 处理 "倒数"、"最后"、"最右"
        if "倒数" in text or "最后" in text or "最右" in text:
            reverse = True
            text = text.replace("倒数", "").replace("最后", "").replace("最右", "").replace("边", "").replace("面", "")

        # 2. 处理 "第X个/棵/只"
        match = re.search(r'第([一二两三四五六七八九十\d]+)[个棵只条辆位]', text)
        if match:
            num_str = match.group(1)
            if num_str.isdigit():
                idx_val = int(num_str) - 1
            else:
                idx_val = CommandParser.CN_NUM.get(num_str, 0)
            index = max(0, idx_val)
            split_idx = match.end()
            target_obj = text[split_idx:].strip()
        else:
            target_obj = text

        target_obj = target_obj.replace("的", "").strip()
        if not target_obj: target_obj = text

        return target_obj, index, reverse


# ==========================================
# 1. 基础工具 & FPS 控制器
# ==========================================
class VideoFPSController:
    def __init__(self, cap):
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0
        self.target_interval = 1.0 / self.fps

    def sync(self, start_time):
        process_duration = time.time() - start_time
        wait_time_sec = self.target_interval - process_duration
        wait_ms = max(1, int(wait_time_sec * 1000))
        return wait_ms


# ==========================================
# 2. Qwen-VL 封装器 (恢复你的原始逻辑)
# ==========================================
class QwenWrapper:
    def __init__(self, model_path):
        print(f"[System] Loading Qwen Model from: {model_path} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # 【关键修正】恢复使用 Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype="auto",  # 保持auto，让它自动读取config
                device_map="auto",
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_path)
            print("[System] Qwen Model Loaded.")
        except Exception as e:
            print(f"[Error] Load failed: {e}")
            raise e

    def _load_image(self, image_input, max_size=1024):
        # 保持你的 Resize 逻辑不变
        if isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.open(image_input).convert("RGB")

        w, h = pil_img.size
        if w > max_size or h > max_size:
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return pil_img

    def detect_objects(self, image_input, target_text):
        image = self._load_image(image_input)

        # 提示词微调：强制要求检测所有目标，以便我们做排序
        prompt_text = (
            f"Detect all '{target_text}' in the image. "
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)  # 增加token长度以容纳多个目标

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        results = []
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
                    x1 = x1_n * orig_w / 1000
                    y1 = y1_n * orig_h / 1000
                    x2 = x2_n * orig_w / 1000
                    y2 = y2_n * orig_h / 1000
                    bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    results.append({"label": target_text, "bbox": bbox})

        except Exception as e:
            # print(f"[Warning] JSON parse failed: {e}, text: {output_text}")
            # 兼容正则解析
            pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"  # Qwen3 可能的格式
            matches = re.findall(pattern, output_text)
            if not matches:
                # 尝试另一种格式
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
        image = self._load_image(image_input)
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
        return output_text


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

                # 这里只传名词去检测，比如 "树"
                results = self.model.detect_objects(frame, prompt)

                self.output_queue.put((results, capture_time))
                while self.output_queue.qsize() > 1:
                    self.output_queue.get()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Worker Error] {e}")

    def submit_task(self, frame, prompt):
        if self.input_queue.empty():
            self.input_queue.put((frame.copy(), prompt, time.time()))

    def get_result(self):
        if not self.output_queue.empty(): return self.output_queue.get()
        return None

    def stop(self):
        self.running = False
        self.thread.join()


# ==========================================
# 5. 混合追踪器 (集成指令解析与多目标逻辑)
# ==========================================
class HybridTracker:
    def __init__(self, model, correction_interval=1.0):
        self.worker = AsyncQwenWorker(model)
        self.correction_interval = correction_interval
        self.traj_buffer = TrajectoryBuffer(max_duration=4.0)
        self.tracker = None
        self.is_tracking = False
        self.current_bbox = None

        # 状态保存
        self.original_cmd = None  # 用户输入 "第二棵树"
        self.target_noun = None  # 提取名词 "树"
        self.last_submit_time = 0
        self.smooth_factor = 0.3

    def _xyxy_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    def _calculate_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def _smooth_bbox(self, old_box, new_box, factor):
        ox, oy, ow, oh = old_box
        nx, ny, nw, nh = new_box
        rx = int(ox * (1 - factor) + nx * factor)
        ry = int(oy * (1 - factor) + ny * factor)
        rw = int(ow * (1 - factor) + nw * factor)
        rh = int(oh * (1 - factor) + nh * factor)
        return (rx, ry, rw, rh)

    def init_smart(self, frame, user_input):
        """
        初始化逻辑：解析指令 -> 全量检测 -> 逻辑选择
        """
        self.original_cmd = user_input

        # 1. 解析
        target_noun, target_index, is_reverse = CommandParser.parse(user_input)
        self.target_noun = target_noun

        direction_str = "从右向左" if is_reverse else "从左向右"
        print(f"[Init] 解析: '{user_input}' -> 找 '{target_noun}', 顺序: {direction_str}, 索引: {target_index}")

        # 2. 检测
        results = self.worker.model.detect_objects(frame, target_noun)

        if not results:
            print(f"[Init] 未在画面中找到 '{target_noun}'.")
            return False, None

        # 3. 排序 (核心逻辑)
        def get_center_x(res):
            b = res['bbox']
            return (b[0] + b[2]) / 2

        # 根据是否倒序进行排序
        sorted_results = sorted(results, key=get_center_x, reverse=is_reverse)

        print(f"[Init] 找到 {len(sorted_results)} 个 '{target_noun}'.")

        # 4. 选择
        if target_index >= len(sorted_results):
            print(f"[Warning] 索引 {target_index} 越界，已自动选择最后一个。")
            target = sorted_results[-1]
        else:
            target = sorted_results[target_index]

        # 5. 锁定
        bbox_xywh = self._xyxy_to_xywh(target['bbox'])
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox_xywh)
        self.is_tracking = True
        self.current_bbox = bbox_xywh
        self.last_submit_time = time.time()
        self.traj_buffer.add(bbox_xywh)

        return True, bbox_xywh

    def update(self, frame):
        if not self.target_noun:
            return False, None, "Idle", 0

        t_start = time.perf_counter()
        status = "Tracking"

        # 1. CSRT 追踪
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

        # 2. Qwen 异步修正
        async_res = self.worker.get_result()
        if async_res:
            results_list, capture_timestamp = async_res
            if results_list:

                # 在所有结果中，寻找与当前追踪目标空间位置重叠度(IOU)最高的
                best_match_bbox = None
                max_iou = -1

                if self.is_tracking and self.current_bbox:
                    dx, dy = self.traj_buffer.get_delta(capture_timestamp, self.current_bbox)

                    for res in results_list:
                        det_xywh = self._xyxy_to_xywh(res['bbox'])
                        # 补偿位置
                        comp_det_xywh = (det_xywh[0] + dx, det_xywh[1] + dy, det_xywh[2], det_xywh[3])

                        iou = self._calculate_iou(self.current_bbox, comp_det_xywh)
                        if iou > max_iou:
                            max_iou = iou
                            best_match_bbox = det_xywh

                should_reinit = False
                final_bbox = None

                if self.is_tracking:
                    if best_match_bbox and max_iou > 0.6:
                        # 匹配良好，无需修正
                        pass
                    elif best_match_bbox and max_iou > 0.2:
                        # 有偏差，进行平滑修正
                        status = "Correcting"
                        should_reinit = True

                        dx, dy = self.traj_buffer.get_delta(capture_timestamp, self.current_bbox)
                        target_pos = (
                            int(best_match_bbox[0] + dx), int(best_match_bbox[1] + dy),
                            best_match_bbox[2], best_match_bbox[3]
                        )
                        final_bbox = self._smooth_bbox(self.current_bbox, target_pos, self.smooth_factor)

                else:  # Lost 状态
                    # 尝试找回最近的
                    if self.current_bbox:
                        cx, cy = self.current_bbox[0], self.current_bbox[1]
                        min_dist = float('inf')
                        closest_box = None

                        for res in results_list:
                            det_xywh = self._xyxy_to_xywh(res['bbox'])
                            dist = (det_xywh[0] - cx) ** 2 + (det_xywh[1] - cy) ** 2
                            if dist < min_dist:
                                min_dist = dist
                                closest_box = det_xywh

                        if closest_box and min_dist < 50000:  # 距离阈值
                            status = "Re-found"
                            should_reinit = True

                            dx, dy = self.traj_buffer.get_delta(capture_timestamp, self.current_bbox)
                            final_bbox = (
                                int(closest_box[0] + dx), int(closest_box[1] + dy),
                                closest_box[2], closest_box[3]
                            )
                            self.is_tracking = True

                if should_reinit and final_bbox:
                    self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(frame, final_bbox)
                    self.current_bbox = final_bbox

        # 3. 提交任务 (使用提取出的名词，如 "树")
        if self.target_noun and (time.time() - self.last_submit_time > self.correction_interval):
            self.worker.submit_task(frame, self.target_noun)
            self.last_submit_time = time.time()

        return self.is_tracking, self.current_bbox, status, (time.perf_counter() - t_start)


# ==========================================
# 6. 主程序
# ==========================================
def run_demo():
    # 你的本地路径
    MODEL_PATH = r"C:\Users\qijiu\.cache\modelscope\hub\models\Qwen\Qwen3-VL-2B-Instruct"
    VIDEO_PATH = "2.mp4"

    if not os.path.exists(VIDEO_PATH):
        print(f"[Error] Video not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[Error] Failed to open video.")
        return

    fps_controller = VideoFPSController(cap)

    try:
        qwen_wrapper = QwenWrapper(MODEL_PATH)
    except Exception as e:
        print(f"[Error] Model load failed: {e}")
        return

    tracker = HybridTracker(qwen_wrapper, correction_interval=1.0)

    ret, frame = cap.read()
    if not ret: return

    print("\n" + "=" * 50)
    print(" QWEN-VL 智能中文追踪器 (修正版) ")
    print(" 示例输入:")
    print("  - 第二棵树")
    print("  - 倒数第一辆车")
    print("  - 最右边的房子")
    print("=" * 50)

    user_cmd = input("请输入要追踪的目标: ").strip()
    if not user_cmd: user_cmd = "第一棵树"

    success, _ = tracker.init_smart(frame, user_cmd)
    if not success:
        print("[Error] 初始化失败.")
        tracker.worker.stop()
        return

    print("[Info] 追踪已开始. 按 'q' 退出.")

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret: break

            display_frame = frame.copy()
            is_tracking, bbox, status, lat = tracker.update(frame)

            if is_tracking and bbox:
                x, y, w, h = [int(v) for v in bbox]

                color = (0, 255, 0)
                if "Correcting" in status: color = (0, 255, 255)
                if "Re-found" in status: color = (255, 0, 255)

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                info = f"{tracker.original_cmd} | {status} | Lat: {lat * 1000:.0f}ms"
                cv2.putText(display_frame, info, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(display_frame, f"丢失: {tracker.original_cmd}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

            cv2.imshow("Qwen Tracker", display_frame)
            wait_ms = fps_controller.sync(loop_start)
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        tracker.worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()