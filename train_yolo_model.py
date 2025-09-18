#!/usr/bin/env python3
"""
YOLOv8 Custom Training Script for Vihangam Disaster Management System
Train a YOLOv8 model on custom human and debris detection dataset

Usage:
    python train_yolo_model.py

Requirements:
    - ultralytics
    - torch
    - opencv-python
    - matplotlib
    - tensorboard (optional, for advanced logging)
"""

import os
import sys
import time
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VihangamYOLOTrainer:
    def __init__(self, data_config='data.yaml', model_name='yolov8n.pt'):
        """
        Initialize the YOLO trainer for Vihangam disaster management system
        
        Args:
            data_config (str): Path to the data.yaml configuration file
            model_name (str): Pre-trained model to start with
        """
        self.data_config = data_config
        self.model_name = model_name
        self.model = None
        self.training_results = None
        
        # Training configuration
        self.training_config = {
            'epochs': 50,
            'batch': 16,
            'imgsz': 640,
            'patience': 20,
            'save_period': 5,
            'device': 'auto',  # Auto-detect GPU/CPU
            'workers': 8,
            'project': 'vihangam_runs',
            'name': f'disaster_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_period': 5,
            'cache': False,
            'copy_paste': 0.1,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0
        }
        
        logger.info("🚁 Vihangam YOLOv8 Trainer Initialized")
        logger.info(f"📁 Data Config: {data_config}")
        logger.info(f"🤖 Base Model: {model_name}")
        
    def check_environment(self):
        """Check training environment and requirements"""
        logger.info("🔍 Checking training environment...")
        
        # Check if data.yaml exists
        if not os.path.exists(self.data_config):
            logger.error(f"❌ Data configuration file not found: {self.data_config}")
            return False
            
        # Check PyTorch and CUDA
        logger.info(f"🔧 PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 CUDA Available: {gpu_count} GPU(s) detected")
            logger.info(f"💻 Primary GPU: {gpu_name}")
        else:
            logger.info("⚠️  CUDA not available, training will use CPU")
            
        # Check dataset structure
        self.validate_dataset_structure()
        
        return True
        
    def validate_dataset_structure(self):
        """Validate the dataset directory structure"""
        logger.info("📂 Validating dataset structure...")
        
        required_dirs = [
            'images/train',
            'images/val',
            'labels/train', 
            'labels/val'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
                
        if missing_dirs:
            logger.warning(f"⚠️  Missing directories: {missing_dirs}")
            logger.info("📁 Expected directory structure:")
            logger.info("   ├── data.yaml")
            logger.info("   ├── images/")
            logger.info("   │   ├── train/")
            logger.info("   │   └── val/")
            logger.info("   └── labels/")
            logger.info("       ├── train/")
            logger.info("       └── val/")
        else:
            logger.info("✅ Dataset structure validated successfully")
            
        # Count images and labels
        self.count_dataset_files()
        
    def count_dataset_files(self):
        """Count and display dataset statistics"""
        try:
            train_images = len(list(Path('images/train').glob('*.jpg'))) + len(list(Path('images/train').glob('*.png')))
            val_images = len(list(Path('images/val').glob('*.jpg'))) + len(list(Path('images/val').glob('*.png')))
            train_labels = len(list(Path('labels/train').glob('*.txt')))
            val_labels = len(list(Path('labels/val').glob('*.txt')))
            
            logger.info(f"📊 Dataset Statistics:")
            logger.info(f"   📸 Training Images: {train_images}")
            logger.info(f"   🏷️  Training Labels: {train_labels}")
            logger.info(f"   📸 Validation Images: {val_images}")
            logger.info(f"   🏷️  Validation Labels: {val_labels}")
            
            if train_images != train_labels:
                logger.warning(f"⚠️  Mismatch: {train_images} training images vs {train_labels} labels")
            if val_images != val_labels:
                logger.warning(f"⚠️  Mismatch: {val_images} validation images vs {val_labels} labels")
                
        except Exception as e:
            logger.warning(f"⚠️  Could not count dataset files: {e}")
            
    def initialize_model(self):
        """Initialize the YOLO model"""
        try:
            logger.info(f"🤖 Loading YOLOv8 model: {self.model_name}")
            self.model = YOLO(self.model_name)
            
            # Display model info
            logger.info("📋 Model Information:")
            logger.info(f"   Model Type: {type(self.model.model).__name__}")
            logger.info(f"   Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            logger.info(f"   Trainable Params: {sum(p.numel() for p in self.model.model.parameters() if p.requires_grad):,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
            
    def train_model(self):
        """Train the YOLOv8 model"""
        if not self.model:
            logger.error("❌ Model not initialized. Call initialize_model() first.")
            return None
            
        logger.info("🚀 Starting YOLOv8 training for disaster detection...")
        logger.info("=" * 60)
        logger.info(f"📊 Training Configuration:")
        for key, value in self.training_config.items():
            logger.info(f"   {key}: {value}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Start training with progress callback
            self.training_results = self.model.train(
                data=self.data_config,
                **self.training_config
            )
            
            training_time = time.time() - start_time
            logger.info(f"✅ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return None
            
    def display_training_metrics(self):
        """Display and save training metrics"""
        if not self.training_results:
            logger.warning("⚠️  No training results to display")
            return
            
        logger.info("📊 Training Results Summary:")
        logger.info("=" * 60)
        
        # Get results directory
        results_dir = self.training_results.save_dir
        logger.info(f"📁 Results saved to: {results_dir}")
        
        # Try to read results.csv for detailed metrics
        results_file = Path(results_dir) / 'results.csv'
        if results_file.exists():
            import pandas as pd
            try:
                df = pd.read_csv(results_file)
                
                # Display final metrics
                final_metrics = df.iloc[-1]
                logger.info("🎯 Final Training Metrics:")
                
                metric_names = {
                    'train/box_loss': 'Box Loss (Train)',
                    'train/cls_loss': 'Class Loss (Train)', 
                    'train/dfl_loss': 'DFL Loss (Train)',
                    'val/box_loss': 'Box Loss (Val)',
                    'val/cls_loss': 'Class Loss (Val)',
                    'val/dfl_loss': 'DFL Loss (Val)',
                    'metrics/precision(B)': 'Precision',
                    'metrics/recall(B)': 'Recall',
                    'metrics/mAP50(B)': 'mAP@0.5',
                    'metrics/mAP50-95(B)': 'mAP@0.5:0.95'
                }
                
                for csv_col, display_name in metric_names.items():
                    if csv_col in df.columns:
                        value = final_metrics[csv_col]
                        logger.info(f"   {display_name}: {value:.4f}")
                        
            except Exception as e:
                logger.warning(f"⚠️  Could not read detailed metrics: {e}")
        
        # Display model paths
        model_files = {
            'best.pt': 'Best Model',
            'last.pt': 'Last Checkpoint'
        }
        
        logger.info("💾 Saved Models:")
        for filename, description in model_files.items():
            model_path = Path(results_dir) / filename
            if model_path.exists():
                logger.info(f"   {description}: {model_path}")
                
        logger.info("=" * 60)
        
    def validate_trained_model(self):
        """Validate the trained model"""
        if not self.training_results:
            logger.warning("⚠️  No trained model to validate")
            return
            
        try:
            # Load the best model for validation
            best_model_path = Path(self.training_results.save_dir) / 'best.pt'
            
            if best_model_path.exists():
                logger.info("🔍 Running validation on best model...")
                
                # Load and validate
                best_model = YOLO(str(best_model_path))
                val_results = best_model.val(data=self.data_config, verbose=True)
                
                logger.info("✅ Validation completed")
                return val_results
            else:
                logger.warning("⚠️  Best model not found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return None
            
    def save_training_summary(self):
        """Save a summary of the training session"""
        if not self.training_results:
            return
            
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'data_config': self.data_config,
            'training_config': self.training_config,
            'results_directory': str(self.training_results.save_dir),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        summary_file = f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"📄 Training summary saved to: {summary_file}")
        except Exception as e:
            logger.warning(f"⚠️  Could not save training summary: {e}")

def main():
    """Main training function"""
    print("🚁 Vihangam YOLOv8 Disaster Detection Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = VihangamYOLOTrainer(
        data_config='data.yaml',
        model_name='yolov8n.pt'
    )
    
    # Check environment
    if not trainer.check_environment():
        logger.error("❌ Environment check failed. Please fix issues before training.")
        return
        
    # Initialize model
    if not trainer.initialize_model():
        logger.error("❌ Model initialization failed.")
        return
        
    # Train model
    print("\n🚀 Starting training process...")
    print("📊 Training progress will be displayed below:")
    print("-" * 50)
    
    results = trainer.train_model()
    
    if results:
        print("\n" + "=" * 50)
        print("✅ Training completed successfully!")
        
        # Display metrics
        trainer.display_training_metrics()
        
        # Run validation
        trainer.validate_trained_model()
        
        # Save summary
        trainer.save_training_summary()
        
        print("\n🎉 Training session completed!")
        print("📁 Check the 'vihangam_runs' directory for detailed results")
        print("💾 Use 'best.pt' for inference in your disaster management system")
        
    else:
        print("❌ Training failed. Check logs for details.")

if __name__ == "__main__":
    main()