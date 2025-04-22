from ultralytics.utils.plotting import Annotator, colors
from pathlib import Path
import cv2

# Dataset paths
image_dir = Path('/home/lcc/UAVGIT/VisDrone2019/VisDrone2019-DET-val-test/images')
label_dir = Path('/home/lcc/UAVGIT/VisDrone2019/VisDrone2019-DET-val-test/labels')
output_dir = Path('/home/lcc/UAVGIT/VisDrone2019/VisDrone2019-DET-val-test/gt_visualized_nolabel')
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through all images
for image_path in image_dir.glob('*.jpg'):
    label_path = label_dir / (image_path.stem + '.txt')
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️ Skipped: {image_path.name} (couldn't read)")
        continue

    h, w = img.shape[:2]

    # Check if label file exists
    if not label_path.exists():
        print(f"⚠️ Skipped: {image_path.name} (label missing)")
        continue

    with open(label_path, 'r') as f:
        lines = f.read().strip().splitlines()

    annotator = Annotator(img, line_width=1)
    for line in lines:
        cls, x, y, bw, bh = map(float, line.split())
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        box_color = colors(int(cls), True)
        # annotator.box_label([x1, y1, x2, y2], label=str(int(cls)), color=box_color)
        annotator.box_label([x1, y1, x2, y2], color=box_color)

    # Save to output directory
    save_path = output_dir / image_path.name
    cv2.imwrite(str(save_path), annotator.im)

    print(f"✅ Processed: {image_path.name}")

print("🎉 All images processed.")
