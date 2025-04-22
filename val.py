from ultralytics import YOLO
# Load a model
model = YOLO("/home/lcc/UAVGIT/Mamba-YOLO-11/output_dir/mamba_yolo_attention/mambayolo_omni_2442_a2c2fadd_aug180/weights/best.pt")
metrics = model.val(data="/home/lcc/UAVGIT/Mamba-YOLO-11/ultralytics/cfg/datasets/VisDrone.yaml",
                    imgsz=640,
                    batch=16,
                    show_labels=False,)