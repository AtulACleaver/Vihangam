#!/usr/bin/env python3
"""
Quick Training Script for YOLOv8 Custom Dataset
Simple and direct approach for immediate training
"""

from ultralytics import YOLO
import torch
import os
from datetime import datetime

def main():
    print("🚁 Vihangam YOLOv8 Training - Quick Start")
    print("=" * 50)
    
    # Check environment
    print(f"🔧 PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  Using CPU (training will be slower)")
    
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
    
    # Training configuration
    print("\n📊 Training Configuration:")
    print("   Epochs: 50")
    print("   Batch Size: 16")
    print("   Image Size: 640")
    print("   Classes: human, debris")
    
    # Start training
    print("\n🚀 Starting training...")
    print("📈 Training progress will appear below:")
    print("-" * 50)
    
    try:
        # Train the model
        results = model.train(
            data='data.yaml',
            epochs=50,
            imgsz=640,
            batch=16,
            name=f'disaster_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            project='runs/detect',
            patience=15,
            save=True,
            plots=True,
            device='auto',  # Automatically use GPU if available
            verbose=True,
            val=True
        )
        
        print("\n" + "=" * 50)
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        print("💾 Best model: runs/detect/.../weights/best.pt")
        print("📊 Training plots: runs/detect/.../")
        print("\n🎉 Ready to use your custom model!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("💡 Check your dataset structure and try again")

if __name__ == "__main__":
    main()