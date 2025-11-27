import time

import torch
import cv2
import numpy as np
import os

# 尝试导入 DA3，并提供清晰的安装指引
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("[Fatal Error] 'depth_anything_3' 库未找到。")
    print("请通过以下命令安装: pip install git+https://github.com/LiheYoung/Depth-Anything-3-Metric.git")
    # 退出程序或设置一个标志，防止后续代码因缺少依赖而崩溃
    DepthAnything3 = None


class MonocularDepth:
    """
    一个封装了 Depth Anything V3 模型的单目深度估计器。
    - 初始化时加载模型并预热。
    - 提供方法来估计整张图的深度。
    - 提供一个便捷方法直接计算 Bounding Box 内的物体距离。
    """

    def __init__(self, model_type="depth-anything/da3metric-large"):
        if DepthAnything3 is None:
            raise ImportError("DepthAnything V3 库未成功导入，请检查安装。")

        print(f"[Depth] 正在加载 DepthAnything V3 模型 ({model_type})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = DepthAnything3.from_pretrained(model_type).to(self.device).eval()

        # 使用一个标准尺寸的黑色图像进行预热，以避免首次推理的延迟
        print("[Depth] 正在预热模型...")
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.estimate_depth(dummy_frame)  # 调用一次推理
        print("[Depth] 模型加载并预热完毕。")

    def estimate_depth(self, frame_bgr):
        """
        估算单张 BGR 图像的深度图。
        :param frame_bgr: OpenCV BGR 格式的 numpy 数组。
        :return: (raw_depth, vis_img)
                 - raw_depth: 原始深度数据 (float32 numpy 数组, 单位: 米)。
                 - vis_img: 用于显示的可视化深度图 (uint8 BGR numpy 数组)。
        """
        # DA3 需要 RGB 输入
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 模型推理接口接受图像列表
        prediction = self.model.inference([frame_rgb])

        # 提取原始深度数据 (已经是 numpy 数组)
        raw_depth = prediction.depth[0]

        # 创建可视化图像
        vis_img = self._create_visualization(raw_depth)

        return raw_depth, vis_img

    def _create_visualization(self, depth):
        """
        将原始深度数据转换为彩色的可视化图像 (Inferno colormap)。
        """
        # 深度数据已经是 numpy 数组，直接使用
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)  # 处理无效值

        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)

        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    def estimate_bbox_depth(self, frame_bgr, bbox):
        """
        【便捷函数】估算图像中特定边界框内物体的距离。
        :param frame_bgr: OpenCV BGR 格式的 numpy 数组。
        :param bbox: 边界框，格式为 (x, y, w, h)。
        :return: (object_distance, vis_depth_map)
                 - object_distance: 估算的物体距离 (float, 单位: 米)，如果无法计算则为 None。
                 - vis_depth_map: 整张图的可视化深度图。
        """
        # 1. 获取整张图的原始深度数据和可视化图
        raw_depth_map, vis_depth_map = self.estimate_depth(frame_bgr)

        object_distance = None

        # 2. 如果提供了有效的 bbox，则计算其内部深度
        if bbox:
            try:
                x, y, w, h = [int(v) for v in bbox]
                h_map, w_map = raw_depth_map.shape

                # 坐标边界检查
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_map, x + w), min(h_map, y + h)

                # 裁剪出深度图中的感兴趣区域 (ROI)
                if x2 > x1 and y2 > y1:
                    depth_roi = raw_depth_map[y1:y2, x1:x2]

                    # 过滤掉无效值 (如0或离相机太近的点)，然后计算中位数
                    valid_depths = depth_roi[depth_roi > 0.1]
                    if valid_depths.size > 0:
                        # 使用中位数(Median)作为距离估计，它对异常值(如物体边缘)非常稳健
                        object_distance = np.median(valid_depths)
            except Exception as e:
                print(f"[Depth Warning] 计算 BBox 深度时出错: {e}")
                object_distance = None

        return object_distance, vis_depth_map


# =========================================================
# 【内置测试功能】
# 直接运行此脚本 (python monocular_depth.py) 即可进行测试
# =========================================================
if __name__ == "__main__":
    print("--- 正在执行 MonocularDepth 模块的内置测试 ---")

    # 1. 检查测试图片是否存在
    test_image_path = "1.png"
    if not os.path.exists(test_image_path):
        print(f"[Test Error] 测试图片 '{test_image_path}' 不存在，请将其放置在脚本同目录下。")
    else:
        # 2. 初始化深度估计器
        # 这里会加载模型并预热，所以第一次调用会比较慢
        try:
            depth_estimator = MonocularDepth()

            # 3. 读取测试图片
            print(f"\n正在读取测试图片: {test_image_path}")
            image = cv2.imread(test_image_path)

            # 4. 定义一个位于图像中心的选框
            h, w, _ = image.shape
            box_w, box_h = 200, 200
            box_x = (w - box_w) // 2
            box_y = (h - box_h) // 2
            center_bbox = (box_x, box_y, box_w, box_h)
            print(f"在图片中心定义了一个测试选框: {center_bbox}")

            # 5. 调用核心函数进行距离推算
            print("正在调用 estimate_bbox_depth 进行距离推算...")
            start_time = time.time()
            distance, vis_map = depth_estimator.estimate_bbox_depth(image, center_bbox)
            end_time = time.time()

            # 6. 打印和保存结果
            if distance is not None:
                print(f"\n[测试成功] ✔️")
                print(f"  > 估算的中心物体距离为: {distance:.2f} 米")
                print(f"  > 推理耗时: {end_time - start_time:.3f} 秒")
            else:
                print(f"\n[测试失败] ❌")
                print("  >未能计算出距离。")

            # 7. 保存可视化结果以供检查
            output_dir = "test_output"
            os.makedirs(output_dir, exist_ok=True)

            # 在原图上画出测试框
            cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
            cv2.putText(image, f"Test Box", (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            original_with_box_path = os.path.join(output_dir, "original_with_box.png")
            depth_vis_path = os.path.join(output_dir, "depth_visualization.png")

            cv2.imwrite(original_with_box_path, image)
            cv2.imwrite(depth_vis_path, vis_map)

            print(f"\n测试结果已保存至 '{output_dir}' 文件夹:")
            print(f"  - {original_with_box_path} (带有测试框的原图)")
            print(f"  - {depth_vis_path} (深度可视化图)")

        except (ImportError, RuntimeError) as e:
            print(f"\n[Test Error] 初始化或运行时发生错误: {e}")
            print("请确保您的环境已正确安装 PyTorch, CUDA 和 Depth Anything V3。")
        except Exception as e:
            print(f"\n[Test Error] 发生未知错误: {e}")