from ultralytics import YOLO
# Load a model
model = YOLO("/home/lcc/ssd/GitRepo/Mamba-YOLO-11/output_dir/mambayolo_omni_2442_a2c2fadd_aug180/weights/best.onnx")
metrics = model.val(data="/home/lcc/UAVGIT/GitRepo/Mamba-YOLO-11/ultralytics/cfg/datasets/VisDrone.yaml")