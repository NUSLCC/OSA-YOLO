from ultralytics import YOLO
import argparse
import os
import torch

torch.cuda.device_count.cache_clear()

torch.use_deterministic_algorithms(True, warn_only=False)

task_name = 'mambayolo_yolo12_0.33_aug180'

from clearml import Task
task = Task.init(project_name="mamba-yolo-regular", task_name=task_name)

current_path = os.path.abspath(os.getcwd())

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='detect', help='train, val, test, speed or study')
    # Training settings
    parser.add_argument('--model', type=str, default=current_path+'/ultralytics/cfg/models/12/yolo12n.yaml', help='model path(s)')
    parser.add_argument('--data', type=str, default=current_path+'/ultralytics/cfg/datasets/VisDrone.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--cache', default='disk', help='cache images for faster training')
    parser.add_argument('--device', default=[0,1], help='cuda device, i.e. 0 or 0,1 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', type=str, default=current_path+'/output_dir/mamba_yolo_attention', help='save to project/name')
    parser.add_argument('--name', type=str, default=task_name, help='save to project/name')
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', default=True,  help='# Use automatic mixed precision')
    # Val settings
    parser.add_argument('--half', default=False, help='use FP16 half-precision inference')
    parser.add_argument('--dnn', default=False, help='use OpenCV DNN for ONNX inference')
    # Predict settings
    parser.add_argument('--augment', default=True, help='Data augmentation')
    # Export settings
    parser.add_argument('--int8', default=False, help='INT8 quantization')
    parser.add_argument('--simplify', default=False, help='simplify ONNX model')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    # Hyperparameters
    parser.add_argument('--degrees', type=float, default=180, help='Data augmentation')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    args = {
        "task": task,
        "data": opt.data,
        "epochs": opt.epochs,
        "batch": opt.batch,
        "imgsz": opt.imgsz,
        "cache": opt.cache,
        "device": opt.device,
        "workers": opt.workers,
        "project": opt.project,
        "name": opt.name,
        "optimizer": opt.optimizer,
        "amp": opt.amp,
        "half": opt.half,
        "dnn": opt.dnn,
        "augment": opt.augment,
        "int8": opt.int8,
        "simplify": opt.simplify,
        "workspace": opt.workspace,
        "degrees": opt.degrees,
    }
    model_conf = opt.model
    task_type = {
        "train": YOLO(model_conf).train(**args),
        "val": YOLO(model_conf).val(**args),
        # "test": YOLO(model_conf).test(**args),
    }
    task_type.get(task)
