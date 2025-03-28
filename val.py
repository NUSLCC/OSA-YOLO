from ultralytics import YOLO
# Load a model
model = YOLO("/home/lcc/UAVGIT/Mamba-YOLO-11/output_dir/mamba_yolo12/mambayolo12_1v3v3v1v_2a2a2a3x/weights/best.pt")
metrics = model.val(data="/home/lcc/UAVGIT/Mamba-YOLO-11/ultralytics/cfg/datasets/VisDrone.yaml")