#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
from datetime import datetime

def check_dataset():
    data_path = 'data.yaml'
    if not os.path.exists(data_path):
        print(f"Data config not found: {data_path}")
        return False
        
    # Check basic structure
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    missing = [d for d in dirs if not os.path.exists(d)]
    
    if missing:
        print(f"Missing directories: {missing}")
        return False
        
    print("Dataset structure looks good")
    return True

def train_model(epochs=50, batch_size=16, model='yolov8n.pt'):
    if not check_dataset():
        return
        
    print(f"Starting training with {model}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    yolo = YOLO(model)
    
    results = yolo.train(
        data='data.yaml',
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        patience=20,
        device='auto',
        project='runs/train',
        name=f'disaster_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        exist_ok=True
    )
    
    print("Training completed!")
    return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--model', default='yolov8n.pt', help='Base model')
    
    args = parser.parse_args()
    
    print("YOLO Training Script")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}")
    
    results = train_model(args.epochs, args.batch, args.model)
    
    if results:
        print(f"Training completed. Results saved to: {results.save_dir}")
    else:
        print("Training failed.")

if __name__ == "__main__":
    main()
