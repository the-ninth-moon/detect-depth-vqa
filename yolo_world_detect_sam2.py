import os
import time
import torch
import cv2
import numpy as np
import threading
import queue
import sys
import copy

# 第三方库
import translators as ts
from ultralytics import YOLO

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("[Error] 未检测到 SAM 2 库。")
    sys.exit(1)

# ================= 配置区域 =================
# 1. 定义你的项目根目录 (请确认这个路径是你存放代码和 sam2 文件夹的根目录)
PROJECT_ROOT = r"C:\Users\qijiu\Desktop\dj_cloude_demo\detect-depth-vqa"

# 2. 拼接绝对路径 (解决 Hydra 找不到文件的问题)
# 指向: .../sam2/configs/sam2.1/sam2.1_hiera_t.yaml
SAM2_CONFIG = os.path.join(PROJECT_ROOT, "sam2", "configs", "sam2.1", "sam2.1_hiera_t.yaml")

# 指向: .../checkpoints/sam2.1_hiera_tiny.pt
SAM2_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")

# 3. 性能参数
INFERENCE_SIZE = 1024

# 4. 路径检查 (防止路径写错直接报错)
if not os.path.exists(SAM2_CONFIG):
    print(f"[Error] 配置文件未找到: {SAM2_CONFIG}")
    print("请检查 sam2 文件夹是否完整，以及路径是否正确。")
    sys.exit(1)

if not os.path.exists(SAM2_CHECKPOINT):
    print(f"[Error] 权重文件未找到: {SAM2_CHECKPOINT}")
    sys.exit(1)
# ===========================================


# ===========================================

class Translator:
    def __init__(self):
        pass

    def translate(self, text, target_lang='en'):
        if not text: return ""
        try:
            return ts.translate_text(query_text=text, to_language=target_lang)
        except:
            return text


class YOLOWorldWrapper:
    def __init__(self, model_path, conf_threshold=0.1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        print(f"[System] YOLO-World loaded on {self.device}")

    def detect_objects(self, image, text):
        # image 应该是原图
        self.model.set_classes([text])
        results = self.model.predict(image, conf=self.conf, device=self.device, verbose=False, imgsz=640)
        formatted = []
        if len(results) > 0:
            for box in results[0].boxes.cpu().numpy():
                formatted.append({
                    "bbox": box.xyxy[0].tolist(),  # x1, y1, x2, y2
                    "conf": box.conf[0]
                })
        return formatted


class AsyncDetectorWorker:
    def __init__(self, model):
        self.model = model
        self.queue_in = queue.Queue(maxsize=1)
        self.queue_out = queue.Queue(maxsize=1)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                frame, prompt = self.queue_in.get(timeout=0.1)
                res = self.model.detect_objects(frame, prompt)
                if self.queue_out.full(): self.queue_out.get_nowait()
                self.queue_out.put(res)
            except:
                continue

    def submit(self, frame, prompt):
        if not self.queue_in.full():
            self.queue_in.put((frame.copy(), prompt))  # Copy很重要，防止多线程竞争

    def get_result(self):
        try:
            return self.queue_out.get_nowait()
        except:
            return None

    def stop(self):
        self.running = False


class FastSAM2Tracker:
    def __init__(self, detector):
        self.worker = AsyncDetectorWorker(detector)
        self.translator = Translator()

        print("[System] Loading SAM 2 (BFloat16 Optimized)...")
        self.device = "cuda"
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

        self.is_tracking = False
        self.last_box = None
        self.target_prompt = ""
        self.scale_factor = 1.0
        self.lost_counter = 0
        self.last_yolo_time = 0

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = INFERENCE_SIZE / max(h, w)
        if scale >= 1: return frame, 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        return resized, scale

    # ==================================================
    # 核心修改：带“保底机制”的手动初始化
    # ==================================================
    def init_with_box(self, frame, box_xywh):
        """
        手动初始化：优先用 SAM 2 精修，如果失败则回退到用户框。
        """
        x, y, w, h = box_xywh
        raw_box = np.array([x, y, x + w, y + h])  # xyxy

        # 1. 预处理
        img_small, scale = self._preprocess(frame)
        self.predictor.set_image(img_small)

        # 2. 坐标缩放
        box_small = raw_box * scale

        # 3. 第一次预测
        # multimask_output=True 有时能给更多选择，但这里我们相信用户的框够准，用 False
        masks, scores, _ = self.predictor.predict(box=box_small, multimask_output=False)

        # 4. 决策逻辑
        # 只要 score > -1.0 (基本都过)，我们就尝试用 mask
        best_mask = masks[0]
        refined_box = self._mask_to_box_clean(best_mask)

        if refined_box is not None:
            # 方案 A: SAM 2 成功识别，使用精修框
            self.last_box = refined_box
            print("[Tracker] Init: SAM 2 Refined Box.")
        else:
            # 方案 B: SAM 2 没看懂，强制使用用户的框 (保底)
            # 这种情况通常发生在物体太小、纹理太弱或者框选背景太多时
            self.last_box = box_small
            print("[Tracker] Init: Fallback to User Box.")

        self.scale_factor = scale
        self.is_tracking = True
        self.target_prompt = "Manual"
        self.lost_counter = 0
        return True

    def init_tracker(self, frame, text):
        self.target_prompt = self.translator.translate(text)
        print(f"[Target] {self.target_prompt}")

        # 1. YOLO 全图检测 (用原图)
        results = self.worker.model.detect_objects(frame, self.target_prompt)
        if not results: return False

        # 2. 逻辑选择 (选第二棵树)
        results.sort(key=lambda x: x['bbox'][0])
        idx = 1 if (len(results) >= 2 and ("第二" in text or "2nd" in text)) else 0
        raw_box = np.array(results[idx]['bbox'])  # xyxy

        # 3. 初始帧 SAM 2 确认
        img_small, scale = self._preprocess(frame)
        self.predictor.set_image(img_small)

        # 将原图坐标缩放到 inference 坐标
        box_small = raw_box * scale
        masks, scores, _ = self.predictor.predict(box=box_small, multimask_output=False)

        if scores[0] > 0.5:
            self.last_box = self._mask_to_box(masks[0])  # 获取紧致框 (inference 坐标)
            self.scale_factor = scale
            self.is_tracking = True
            return True
        return False

    def update(self, frame):
        if not self.is_tracking:
            self._attempt_recovery(frame)
            return False, None, None, "Lost"

        t0 = time.time()

        # 1. 图像缩放
        img_small, scale = self._preprocess(frame)
        self.scale_factor = scale

        # 2. SAM 2 图像编码
        self.predictor.set_image(img_small)

        # =========================================================
        # 【核心修改点：宽进严出策略】
        # =========================================================

        # 3. 计算 Prompt Box (搜索范围) - 宽进
        # 镜头甩动越快，这个 pad 需要越大。
        # 在 1024 尺寸下，30~50 像素通常能覆盖快速甩头
        search_pad = 40

        x1, y1, x2, y2 = self.last_box
        h_img, w_img = img_small.shape[:2]

        # 扩大搜索范围，为了抓住快速移动的物体
        p_x1 = max(0, x1 - search_pad)
        p_y1 = max(0, y1 - search_pad)
        p_x2 = min(w_img, x2 + search_pad)
        p_y2 = min(h_img, y2 + search_pad)

        prompt_box = np.array([p_x1, p_y1, p_x2, p_y2])

        # 4. 预测
        masks, scores, _ = self.predictor.predict(box=prompt_box, multimask_output=False)
        best_mask = masks[0]
        score = scores[0]

        status = "Tracking"

        # 5. 结果更新 - 严出
        # 只有当置信度还可以时，才更新位置
        if score > 0.0:  # SAM2 的 score 有时会偏低，只要 mask 有东西就行
            clean_box = self._mask_to_box_clean(best_mask)

            if clean_box is not None:
                # 【关键逻辑】：
                # 如果新计算的框和上一帧的框距离太远（比如突变到了画面另一端），可能是漂移
                # 这里可以加一个简单的中心点距离校验，防止跳变到背景的干扰物上
                prev_cx = (x1 + x2) / 2
                new_cx = (clean_box[0] + clean_box[2]) / 2

                # 如果中心点移动超过 150 像素 (在小图上)，认为是不合理的跳变，忽略这次更新
                if abs(new_cx - prev_cx) < 150:
                    self.last_box = clean_box  # 更新为紧致框
                    self.lost_counter = 0
                else:
                    # 距离太远，可能跟丢了，保持上一帧位置，等待下一帧（或触发找回）
                    status = "Jump Detected"
                    self.lost_counter += 1
            else:
                self.lost_counter += 1

            display_box = self.last_box / scale
            display_mask = best_mask

        else:
            self.lost_counter += 1
            status = "Low Conf"
            display_box = None
            display_mask = None

        # 连续丢失判定
        if self.lost_counter > 10:  # 给多一点宽容度
            self.is_tracking = False

        # 5. 后台 Re-ID 检查 (每隔 30 帧或 1 秒)
        if time.time() - self.last_yolo_time > 1.0:
            self.worker.submit(frame, self.target_prompt)
            self.last_yolo_time = time.time()

        fps = 1 / (time.time() - t0)
        return True, display_box, display_mask, f"{status} {fps:.0f}FPS"

    def _mask_to_box(self, mask):
        y, x = np.where(mask > 0)
        if len(x) == 0: return self.last_box  # 保持不变
        return np.array([x.min(), y.min(), x.max(), y.max()])

    def _mask_to_box(self, mask):
        # 旧的简单方法，保留备用
        y, x = np.where(mask > 0)
        if len(x) == 0: return None
        return np.array([x.min(), y.min(), x.max(), y.max()])

    def _mask_to_box_clean(self, mask):
        """
        更鲁棒的 Mask 转 Box：
        1. 阈值化处理
        2. 只取最大轮廓
        3. 处理空 Mask 情况
        """
        # SAM2 输出的 mask 可能是 logit 也可能是概率，通常 > 0 表示前景
        mask_uint8 = (mask > 0.0).astype(np.uint8)

        # 查找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None  # 确实没东西

        # 找到面积最大的轮廓 (过滤噪点)
        max_cnt = max(contours, key=cv2.contourArea)

        # 如果最大轮廓太小（比如只是几个像素的噪点），也视为无效
        if cv2.contourArea(max_cnt) < 10:
            return None

        x, y, w, h = cv2.boundingRect(max_cnt)
        return np.array([x, y, x + w, y + h])
    def _attempt_recovery(self, frame):
        res = self.worker.get_result()
        if res and len(res) > 0:
            # 简单策略：取置信度最高的。
            # 进阶：在这里加入你的特征比对 (Re-ID) 逻辑
            print("[System] Attempting recovery via YOLO...")

            best = res[0]['bbox']
            # 立即验证
            img_small, scale = self._preprocess(frame)
            self.predictor.set_image(img_small)
            box_small = np.array(best) * scale
            masks, scores, _ = self.predictor.predict(box=box_small, multimask_output=False)

            if scores[0] > 0.6:
                self.last_box = self._mask_to_box(masks[0])
                self.scale_factor = scale
                self.is_tracking = True
                print("[System] Recovered!")

    def stop(self):
        self.worker.stop()


# ================= 主程序 =================
def run():
    # 替换为你的路径
    YOLO_PATH = 'yolov8s-world.pt'
    VIDEO_PATH = "1.mp4"  # 或 0 (摄像头)

    detector = YOLOWorldWrapper(YOLO_PATH)
    tracker = FastSAM2Tracker(detector)

    cap = cv2.VideoCapture(VIDEO_PATH)

    # 获取第一帧用于初始化
    ret, frame = cap.read()
    if not ret: return

    cmd = input("请输入目标 (默认: 第二棵树): ").strip()
    if not cmd: cmd = "第二棵树"

    tracker.init_tracker(frame, cmd)

    while True:
        ret, frame = cap.read()
        if not ret: break

        ok, box, mask, info = tracker.update(frame)

        # 绘图逻辑
        if ok and box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制 Mask (需要将小 mask 放大回原图)
            if mask is not None:
                # 这里的 mask 是 inference_size 的，需要 resize 回 frame 的大小
                full_mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

                # 快速绘制半透明 mask
                color_mask = np.zeros_like(frame)
                color_mask[full_mask > 0] = (0, 100, 255)
                frame = cv2.addWeighted(frame, 0.8, color_mask, 0.2, 0)

        cv2.putText(frame, info, (20, 40), 0, 1.0, (0, 255, 255), 2)
        cv2.imshow("Real-time Tracking", frame)

        if cv2.waitKey(1) == ord('q'): break

    tracker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()