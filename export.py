from ultralytics import YOLO

model = YOLO("/home/lcc/ssd/GitRepo/Mamba-YOLO-11/output_dir/mambayolo_omni_2442_a2c2fadd_aug180/weights/best.pt")

model.export(format="engine", 
             imgsz=640,
             half=False,
             int8=False,
             simplify=True,
             data="/home/lcc/UAVGIT/GitRepo/Mamba-YOLO-11/ultralytics/cfg/datasets/VisDrone.yaml",
             device=0,
             )