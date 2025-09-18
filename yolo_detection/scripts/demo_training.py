#!/usr/bin/env python3
"""
YOLOv8 Training Demo for Vihangam Disaster Management
Demonstrates the training process with sample data creation
"""

from ultralytics import YOLO
import torch
import os
import numpy as np
from PIL import Image, ImageDraw
import random
from datetime import datetime

def create_sample_images():
    """Create sample training images with basic shapes"""
    print("🖼️  Creating sample training data...")
    
    # Sample images for training and validation
    train_samples = 20
    val_samples = 5
    
    def create_image_with_objects(filename, label_filename, split='train'):
        """Create a sample image with rectangles representing humans and debris"""
        img_size = (640, 480)
        img = Image.new('RGB', img_size, color='lightblue')
        draw = ImageDraw.Draw(img)
        
        labels = []
        num_objects = random.randint(1, 3)
        
        for _ in range(num_objects):
            # Randomly choose class (0=human, 1=debris)
            class_id = random.choice([0, 1])
            
            # Generate random bounding box
            x1 = random.randint(50, img_size[0] - 150)
            y1 = random.randint(50, img_size[1] - 150)
            width = random.randint(50, 100)
            height = random.randint(80, 120)
            x2 = x1 + width
            y2 = y1 + height
            
            # Draw rectangle (different colors for different classes)
            color = 'red' if class_id == 0 else 'brown'
            draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=2)
            
            # Add label text
            label_text = 'Human' if class_id == 0 else 'Debris'
            draw.text((x1, y1-20), label_text, fill='black')
            
            # Convert to YOLO format (normalized)
            center_x = (x1 + x2) / 2 / img_size[0]
            center_y = (y1 + y2) / 2 / img_size[1]
            norm_width = width / img_size[0]
            norm_height = height / img_size[1]
            
            labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Save image and label
        img_path = f'images/{split}/{filename}.jpg'
        label_path = f'labels/{split}/{filename}.txt'
        
        img.save(img_path)
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
        
        return len(labels)
    
    # Create training samples
    total_train_objects = 0
    for i in range(train_samples):
        objects = create_image_with_objects(f'train_img_{i:03d}', f'train_img_{i:03d}', 'train')
        total_train_objects += objects
    
    # Create validation samples  
    total_val_objects = 0
    for i in range(val_samples):
        objects = create_image_with_objects(f'val_img_{i:03d}', f'val_img_{i:03d}', 'val')
        total_val_objects += objects
    
    print(f"✅ Sample dataset created:")
    print(f"   📸 Training: {train_samples} images, {total_train_objects} objects")
    print(f"   📸 Validation: {val_samples} images, {total_val_objects} objects")

def demo_training():
    """Demonstrate YOLOv8 training process"""
    print("🚁 Vihangam YOLOv8 Training Demo")
    print("=" * 50)
    
    # Check environment
    print(f"🔧 PyTorch: {torch.__version__}")
    print("💻 Using CPU for demo (faster setup)")
    
    # Create sample dataset
    create_sample_images()
    
    # Check data.yaml
    if not os.path.exists('data.yaml'):
        print("❌ data.yaml not found!")
        print("📁 Please ensure data.yaml exists with your dataset configuration")
        return
    
    print("✅ data.yaml found")
    
    # Initialize model
    print("\n🤖 Loading YOLOv8n model...")
    try:
        model = YOLO('yolov8n.pt')
        print("✅ Model loaded successfully")
        
        # Show model architecture info
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"📊 Model Stats: {total_params:,} total parameters ({trainable_params:,} trainable)")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Training configuration for demo (short training)
    print("\n📊 Demo Training Configuration:")
    print("   Epochs: 5 (demo - normally use 50+)")
    print("   Batch Size: 2 (small for demo)")
    print("   Image Size: 320 (reduced for speed)")
    print("   Classes: human, debris")
    
    # Start training
    print("\n🚀 Starting demo training...")
    print("📈 Training progress:")
    print("-" * 50)
    
    try:
        # Train the model with demo settings (very short)
        results = model.train(
            data='data.yaml',
            epochs=5,            # Very short for demo
            imgsz=320,           # Smaller for speed
            batch=2,             # Small batch
            name=f'disaster_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            project='runs/detect',
            patience=10,
            save=True,
            plots=True,
            device='cpu',
            verbose=True,
            val=True,
            workers=0,
            cache=False,
            amp=False,
        )
        
        print("\n" + "=" * 50)
        print("✅ Demo training completed!")
        print(f"📁 Results saved to: {results.save_dir}")
        
        # Test the trained model
        print("\n🔍 Testing the trained model...")
        
        # Load the best model and run a quick test
        best_model = YOLO(f'{results.save_dir}/weights/best.pt')
        
        # Test on a sample image
        test_results = best_model('images/val/val_img_000.jpg', verbose=False)
        
        if test_results and len(test_results) > 0:
            detections = test_results[0].boxes
            if detections is not None:
                print(f"🎯 Test detection: Found {len(detections)} objects")
                for i, detection in enumerate(detections):
                    conf = float(detection.conf[0])
                    cls = int(detection.cls[0])
                    class_name = 'human' if cls == 0 else 'debris'
                    print(f"   Object {i+1}: {class_name} (confidence: {conf:.2f})")
            else:
                print("🎯 Test detection: No objects found")
        
        # Show what was created
        print(f"\n💾 Training artifacts created:")
        print(f"   📦 Best model: {results.save_dir}/weights/best.pt")
        print(f"   📦 Last checkpoint: {results.save_dir}/weights/last.pt")
        print(f"   📊 Training plots: {results.save_dir}/*.png")
        print(f"   📈 Metrics: {results.save_dir}/results.csv")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 For production training:")
        print("   1. Use real disaster images (hundreds/thousands)")
        print("   2. Train for 50-100 epochs") 
        print("   3. Use larger batch sizes (8-32)")
        print("   4. Use GPU if available")
        print("   5. Fine-tune hyperparameters")
        
        print("\n🔧 Integration with Vihangam:")
        print(f"   # Update your yolo_handler.py:")
        print(f"   model_path = '{results.save_dir}/weights/best.pt'")
        print(f"   detector = YOLOHandler(model_path)")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Demo training failed: {e}")
        print("💡 Common issues:")
        print("   - Insufficient disk space")
        print("   - Memory constraints") 
        print("   - Missing dependencies")
        return None

if __name__ == "__main__":
    # Run the demo
    results = demo_training()
    
    if results:
        print("\n" + "="*60)
        print("🎓 TRAINING COMPLETE - NEXT STEPS")
        print("="*60)
        print("1. 📁 Review training plots in the results directory")
        print("2. 🧪 Test your model on new images")
        print("3. 🔄 Integrate the best.pt model into your Django app")
        print("4. 🚀 Deploy your custom disaster detection system!")
        print("="*60)