import collections
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
    import torch._six
except ImportError:
    # 如果找不到 torch._six，就手动创建一个假的
    class _Six:
        string_classes = str
        container_abcs = collections.abc
        int_classes = int


    # 将其注入到系统模块中
    sys.modules['torch._six'] = _Six()
    torch._six = _Six()
    print("[System] PyTorch 2.x Compatibility Patch Applied (torch._six)")
# ==========================================================
# ================= 配置区域 =================
# 1. 定义项目根目录
PROJECT_ROOT = r"C:\Users\qijiu\Desktop\dj_cloude_demo\detect-depth-vqa"

# 2. 关键：将 OSTrack 的 lib 路径加入系统路径，否则无法导入
LIB_PATH = os.path.join(PROJECT_ROOT, "lib")
print(LIB_PATH)
sys.path.append(LIB_PATH)

# 3. OSTrack 权重路径
OSTRACK_WEIGHT = os.path.join(PROJECT_ROOT, "checkpoints", "ostrack_vitb_256_mae_ce_32x4_ep300.pth.tar")

# 4. 检查路径
if not os.path.exists(LIB_PATH):
    print(f"[Error] 未找到 lib 文件夹: {LIB_PATH}")
    print("请从 OSTrack 官方仓库下载代码，并将 lib 文件夹复制到项目根目录。")
    sys.exit(1)

if not os.path.exists(OSTRACK_WEIGHT):
    print(f"[Error] 权重文件未找到: {OSTRACK_WEIGHT}")
    sys.exit(1)

# 5. 导入 OSTrack 模块 (需要在 sys.path 添加后导入)
try:
    from lib.test.evaluation.tracker import Tracker
except ImportError as e:
    print(f"[Error] 无法导入 OSTrack 模块: {e}")
    print("请检查 lib/test/evaluation/tracker.py 是否存在。")
    sys.exit(1)


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
            self.queue_in.put((frame.copy(), prompt))

    def get_result(self):
        try:
            return self.queue_out.get_nowait()
        except:
            return None

    def stop(self):
        self.running = False


try:
    from lib.test.tracker.ostrack import OSTrack
    from lib.config.ostrack.config import cfg
except ImportError:
    print("[Error] 无法导入 OSTrack 核心类。")
    sys.exit(1)

class OSTrackWrapper:
    def __init__(self, detector):
        self.worker = AsyncDetectorWorker(detector)
        self.translator = Translator()

        print("[System] Loading OSTrack (Native Mode - 256x256)...")

        # 1. 准备参数容器
        class TrackerParams:
            def __init__(self):
                self.cfg = cfg
                self.checkpoint = OSTRACK_WEIGHT
                self.debug = False
                self.save_all_boxes = False

                # 推理必须的参数属性
                self.template_factor = 2.0
                self.template_size = 128
                self.search_factor = 4.0
                self.search_size = 256

        self.params = TrackerParams()

        # 2. 配置模型参数 (Config)
        common_cfg = self.params.cfg

        # (1) 尺寸参数 (必须匹配权重)
        common_cfg.DATA.SEARCH.SIZE = 256
        common_cfg.DATA.TEMPLATE.SIZE = 128

        # (2) 模型结构参数
        common_cfg.MODEL.VIT_TYPE = 'vit_base_patch16_224'
        common_cfg.MODEL.BACKBONE.STRIDE = 16
        common_cfg.MODEL.BACKBONE.CE_LOC = [3, 6, 9]
        common_cfg.MODEL.BACKBONE.CE_KEEP_RATIO = [0.7, 0.7, 0.7]

        # 【关键修复】这里必须是 'CTR_POINT'，不能是 'ON'
        common_cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'CTR_POINT'

        common_cfg.MODEL.HEAD_TYPE = "CENTER"
        common_cfg.MODEL.HIDDEN_DIM = 768

        # (3) 推理参数同步
        common_cfg.TEST.SEARCH_FACTOR = 4.0
        common_cfg.TEST.SEARCH_SIZE = 256
        common_cfg.TEST.TEMPLATE_FACTOR = 2.0
        common_cfg.TEST.TEMPLATE_SIZE = 128

        # 3. 初始化跟踪器
        self.tracker = OSTrack(self.params, dataset_name='got10k')

        self.is_tracking = False
        self.target_prompt = ""
        self.last_yolo_time = 0
        self.last_box_xyxy = None

    def init_with_box(self, frame, box_xywh):
        """ 手动初始化 """
        init_info = {'init_bbox': box_xywh}
        self.tracker.initialize(frame, init_info)

        x, y, w, h = box_xywh
        self.last_box_xyxy = [x, y, x + w, y + h]
        self.is_tracking = True
        self.target_prompt = "Manual"
        print("[Tracker] Init: OSTrack Initialized Manual.")
        return True

    def init_tracker(self, frame, text):
        """ YOLO -> OSTrack 初始化 """
        self.target_prompt = self.translator.translate(text)
        print(f"[Target] {self.target_prompt}")

        # YOLO 全图检测
        results = self.worker.model.detect_objects(frame, self.target_prompt)
        if not results: return False

        # 逻辑选择
        results.sort(key=lambda x: x['bbox'][0])
        idx = 1 if (len(results) >= 2 and ("第二" in text or "2nd" in text)) else 0
        raw_box_xyxy = results[idx]['bbox']

        # 转换坐标 xyxy -> xywh
        x1, y1, x2, y2 = raw_box_xyxy
        w = x2 - x1
        h = y2 - y1
        box_xywh = [x1, y1, w, h]

        # OSTrack 初始化
        self.tracker.initialize(frame, {'init_bbox': box_xywh})

        self.last_box_xyxy = raw_box_xyxy
        self.is_tracking = True
        print("[Tracker] Init: OSTrack Initialized via YOLO.")
        return True

    def update(self, frame):
        if not self.is_tracking:
            self._attempt_recovery(frame)
            return False, None, None, "Lost"

        t0 = time.time()

        # OSTrack 推理
        out = self.tracker.track(frame)
        res_bbox = out['target_bbox']  # [x, y, w, h]

        # 转换回 xyxy 供绘图
        x, y, w, h = res_bbox
        self.last_box_xyxy = [x, y, x + w, y + h]

        fps = 1 / (time.time() - t0)

        # 简单的防漂移检查
        if time.time() - self.last_yolo_time > 1.0 and self.target_prompt != "Manual":
            self.worker.submit(frame, self.target_prompt)
            self.last_yolo_time = time.time()

        return True, self.last_box_xyxy, None, f"OSTrack {fps:.0f}FPS"

    def _attempt_recovery(self, frame):
        res = self.worker.get_result()
        if res and len(res) > 0:
            print("[System] Attempting recovery via YOLO...")
            best_box = res[0]['bbox']
            x1, y1, x2, y2 = best_box
            w = x2 - x1
            h = y2 - y1
            self.tracker.initialize(frame, {'init_bbox': [x1, y1, w, h]})
            self.is_tracking = True
            print("[System] Recovered!")

    def stop(self):
        self.worker.stop()

# ================= 主程序 =================
def run():
    # 替换为你的路径
    YOLO_PATH = 'yolov8s-world.pt'
    VIDEO_PATH = "1.mp4"

    # 检查模型是否存在
    if not os.path.exists(YOLO_PATH):
        # 自动下载一个小模型用于测试
        print("未找到 YOLO 模型，将使用 yolov8s-worldv2.pt")
        YOLO_PATH = 'yolov8s-worldv2.pt'

    detector = YOLOWorldWrapper(YOLO_PATH)
    # 替换为 OSTrackWrapper
    tracker = OSTrackWrapper(detector)

    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame = cap.read()
    if not ret: return

    cmd = input("请输入目标 (默认: 第二棵树): ").strip()
    if not cmd: cmd = "第二棵树"

    # 初始化
    success = tracker.init_tracker(frame, cmd)
    if not success:
        print("初始帧未检测到目标，请手动框选或重启。")
        # 这里也可以接一个 cv2.selectROI 的逻辑作为保底

    while True:
        ret, frame = cap.read()
        if not ret: break

        ok, box, mask, info = tracker.update(frame)

        # 绘图逻辑
        if ok and box is not None:
            x1, y1, x2, y2 = map(int, box)
            # OSTrack 只有框，画绿色矩形
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 显示状态
        cv2.putText(frame, info, (20, 40), 0, 1.0, (0, 255, 255), 2)
        cv2.imshow("OSTrack Tracking", frame)

        if cv2.waitKey(1) == ord('q'): break

    tracker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()