
"""
# conda环境，python3.8 

# 安装torch， https://pytorch.org/get-started/locally/
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# 安装tensorrt，https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt==8.0.1.6

# 安装torch2trt，https://github.com/NVIDIA-AI-IOT/torch2trt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install

# 安装opencv
pip install opencv-contrib-python

# 安装 albumentations
pip install albumentations

# 安装cuda
https://developer.nvidia.com/cuda-11-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local
"""

import torch
from torch2trt import torch2trt, TRTModule
import tensorrt as trt
import cv2

from infer import QDNetModel, get_transforms

def pth2trt(model_pth, torch2trtPath, fp16, image_size=320):
	print("torch2trt, may take 1 minute...")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	x = torch.ones(1, 3, image_size, image_size).to(device)
	model_pth.float()
	model_trt = torch2trt(model_pth, [x], fp16_mode=fp16, 
					log_level=trt.Logger.INFO, 
					max_workspace_size=(1 << 32),)
	#能被torch2trt.TRTModule导入的pytorch模型
	pred = model_trt(x)
	torch.save(model_trt.state_dict(), torch2trtPath) 

class QDNetModel_Trt(QDNetModel):
	def __init__(self, torch2trtPath, image_size=256):
		self.transforms_val = get_transforms(image_size)
		self.model = TRTModule()
		self.model.load_state_dict(torch.load(torch2trtPath))

torch2trtPath = "./model1.torch2trt"
infer_obj = QDNetModel()
model_pth = infer_obj.modeltorch2trt
# pth2trt(model_pth, torch2trtPath, fp16=1, image_size=256)

infer_obj_trt = QDNetModel_Trt(torch2trtPath, image_size=256)
img1 = cv2.imread("./finance.png")
l1 = infer_obj.predict(img1)
l2 = infer_obj_trt.predict(img1)
print(l1, l2)

