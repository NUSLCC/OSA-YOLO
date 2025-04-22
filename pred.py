from ultralytics import YOLO
# Load a model
model = YOLO("/home/lcc/UAVGIT/Mamba-YOLO-11/output_dir/mamba_yolo_attention/mambayolo_omni_2442_a2c2fadd_aug180/weights/best.pt")
# results = model("/home/lcc/UAVGIT/VisDrone2019/VisDrone2019-DET-val-test/images")
# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk
# Run inference on 'bus.jpg' with arguments
model.predict(source = "/home/lcc/UAVGIT/VisDrone2019/VisDrone2019-DET-val-test/images", 
              conf=0.25,
              imgsz=640,
              device="0",
              batch=32,
              save=True, 
              show_labels=False,
            #   font_size=8,
            #   show_conf=True,
              show_boxes=True,
              line_width=1,
            #   filename="/home/lcc/UAVGIT/VisDrone2019/VisDrone2019-DET-val-result",
              )