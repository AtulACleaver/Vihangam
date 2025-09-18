#!/usr/bin/env python3
"""
CPU-Optimized YOLOv8 Training Script for Vihangam
Trains on CPU with optimized settings
"""

from ultralytics import YOLO
import torch
import os
from datetime import datetime

def main():
    print("🚁 Vihangam YOLOv8 CPU Training")
    print("=" * 50)
    
    # Check environment
    print(f"🔧 PyTorch: {torch.__version__}")
    print("💻 Using CPU for training")
    
    # Check data.yaml
    if not os.path.exists('data.yaml'):
        print("❌ data.yaml not found!")
        print("📁 Make sure you have the data.yaml file in the current directory")
        return
    
    print("✅ data.yaml found")
    
    # Initialize model
    print("\n🤖 Loading YOLOv8n model...")
    try:
        model = YOLO('yolov8n.pt')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # CPU-optimized training configuration
    print("\n📊 CPU-Optimized Training Configuration:")
    print("   Epochs: 50")
    print("   Batch Size: 4 (reduced for CPU)")
    print("   Image Size: 416 (reduced for CPU)")
    print("   Workers: 0 (CPU optimized)")
    print("   Classes: human, debris")
    
    # Create runs directory if it doesn't exist
    os.makedirs('runs/detect', exist_ok=True)
    
    # Start training
    print("\n🚀 Starting CPU training...")
    print("📈 Training progress (this will take longer on CPU):")
    print("-" * 50)
    
    try:
        # Train the model with CPU-optimized settings
        results = model.train(
            data='data.yaml',
            epochs=50,
            imgsz=416,           # Smaller image size for CPU
            batch=4,             # Smaller batch size for CPU
            name=f'disaster_detection_cpu_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            project='runs/detect',
            patience=25,         # Increased patience for CPU training
            save=True,
            plots=True,
            device='cpu',        # Explicitly use CPU
            verbose=True,
            val=True,
            workers=0,           # Reduce workers for CPU
            cache=False,         # Disable caching on CPU
            amp=False,           # Disable AMP on CPU
            cos_lr=True,         # Cosine learning rate
            close_mosaic=10,     # Close mosaic augmentation
            mixup=0.0,           # Disable mixup for CPU
            copy_paste=0.0,      # Disable copy-paste for CPU
            degrees=10,          # Rotation augmentation
            translate=0.1,       # Translation augmentation
            scale=0.5,           # Scale augmentation
            shear=0.0,           # No shear augmentation
            perspective=0.0,     # No perspective augmentation
            flipud=0.0,          # No vertical flip
            fliplr=0.5,          # Horizontal flip
            mosaic=1.0,          # Mosaic augmentation
            hsv_h=0.015,         # Hue augmentation
            hsv_s=0.7,           # Saturation augmentation
            hsv_v=0.4,           # Value augmentation
        )
        
        print("\n" + "=" * 50)
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        
        # List the created files
        import glob
        weight_files = glob.glob(str(results.save_dir) + "/weights/*.pt")
        if weight_files:
            print("💾 Model files created:")
            for weight_file in weight_files:
                file_size = os.path.getsize(weight_file) / (1024*1024)  # MB
                print(f"   📦 {os.path.basename(weight_file)}: {file_size:.1f} MB")
        
        plot_files = glob.glob(str(results.save_dir) + "/*.png")
        if plot_files:
            print("📊 Training plots created:")
            for plot_file in plot_files[:3]:  # Show first 3 plots
                print(f"   📈 {os.path.basename(plot_file)}")
        
        print("\n🎉 Your custom disaster detection model is ready!")
        print("💡 Integration tips:")
        print("   1. Use 'best.pt' for highest accuracy")
        print("   2. Use 'last.pt' for latest checkpoint")
        print("   3. Update your Django app's YOLO handler to use the custom model")
        
        # Show how to use the model
        print("\n🔧 Usage in your Django app:")
        print(f"   detector = get_yolo_detector('{results.save_dir}/weights/best.pt')")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("💡 Tips:")
        print("   - Check that your dataset images and labels exist")
        print("   - Verify data.yaml paths are correct") 
        print("   - Ensure you have enough disk space")

if __name__ == "__main__":
    main()