import torch
from ultralytics import YOLO

print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name())  # Should show "Orin"

# Test YOLO with CUDA
model = YOLO("yolov8n.yaml").cuda()
print(model.device)  # Should show CUDA device