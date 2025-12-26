import os

class EnvSettings:
    def __init__(self):
        # 项目根目录
        self.prj_dir = r"C:\Users\qijiu\Desktop\dj_cloude_demo\detect-depth-vqa"
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
