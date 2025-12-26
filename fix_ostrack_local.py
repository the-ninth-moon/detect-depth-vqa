import os

# 1. 确定文件要保存的位置
# 目标路径: lib/test/evaluation/local.py
current_dir = os.getcwd()
target_dir = os.path.join(current_dir, "lib", "test", "evaluation")
target_file = os.path.join(target_dir, "local.py")

# 2. 确保目录存在
if not os.path.exists(target_dir):
    print(f"[Error] 找不到目录: {target_dir}")
    print("请确认你在项目根目录下运行此脚本，且 'lib' 文件夹存在。")
    exit(1)

# 3. 定义 local.py 的完整内容 (已修复: 添加了 local_env_settings 函数)
content = f"""import os

class EnvSettings:
    def __init__(self):
        # 项目根目录
        self.prj_dir = r"{current_dir}"
        self.save_dir = os.path.join(self.prj_dir, 'output')
        self.results_path = os.path.join(self.save_dir, 'test/tracking_results')
        self.segmentation_path = os.path.join(self.save_dir, 'test/segmentation_results')

        # 关键：模型权重文件夹路径
        self.network_path = os.path.join(self.prj_dir, 'checkpoints')

        self.result_plot_path = os.path.join(self.save_dir, 'test/result_plots')

        # 数据集路径 (推理时留空即可)
        self.otb_path = ''
        self.got10k_path = ''
        self.lasot_path = ''
        self.trackingnet_path = ''
        self.davis_dir = ''
        self.vot_path = ''
        self.youtubevos_dir = ''

# ==============================================================
# 必需的接口函数
# ==============================================================
def local_env_settings():
    return EnvSettings()
"""

# 4. 写入文件
try:
    with open(target_file, "w", encoding='utf-8') as f:
        f.write(content)
    print(f"[Success] 配置文件已成功修复: {target_file}")
    print("现在你可以重新运行 yolo_world_detect.py 了。")
except Exception as e:
    print(f"[Error] 写入文件失败: {e}")