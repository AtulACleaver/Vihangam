#!/usr/bin/env python3
"""
YOLOv8 Model Validation Script for Vihangam Disaster Management System
Comprehensive performance evaluation of custom trained models
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
from datetime import datetime

from ultralytics import YOLO
import torch

class VihangamModelValidator:
    def __init__(self, model_path=None, data_config='data.yaml'):
        """
        Initialize model validator
        
        Args:
            model_path: Path to the trained model (best.pt)
            data_config: Path to data.yaml configuration
        """
        self.model_path = self.find_best_model_path(model_path)
        self.data_config = data_config
        self.model = None
        self.validation_results = None
        
        print("🚁 Vihangam YOLOv8 Model Validator")
        print("=" * 50)
        print(f"📦 Model: {self.model_path}")
        print(f"📁 Data Config: {self.data_config}")
        
    def find_best_model_path(self, provided_path):
        """Find the best model file"""
        # Check provided path first
        if provided_path and os.path.exists(provided_path):
            return provided_path
            
        # Check common locations
        possible_paths = [
            "runs/detect/train/weights/best.pt",
            "runs/detect/disaster_demo_20250918_201832/weights/best.pt",
            "custom_models/vihangam_disaster_detection.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found model at: {path}")
                return path
                
        # Search for any best.pt in runs directory
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            best_models = list(runs_dir.rglob("best.pt"))
            if best_models:
                latest_model = max(best_models, key=lambda p: p.stat().st_mtime)
                print(f"✅ Found latest model: {latest_model}")
                return str(latest_model)
        
        raise FileNotFoundError("❌ No trained model found! Please provide valid model path.")
    
    def validate_environment(self):
        """Validate validation environment"""
        print("\n🔍 Environment Validation:")
        print("-" * 30)
        
        # Check model file
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            return False
            
        model_size = os.path.getsize(self.model_path) / (1024*1024)
        print(f"📦 Model Size: {model_size:.1f} MB")
        
        # Check data config
        if not os.path.exists(self.data_config):
            print(f"❌ Data config not found: {self.data_config}")
            return False
            
        print(f"✅ Data Config: {self.data_config}")
        
        # Check validation dataset
        val_images_dir = Path("images/val")
        val_labels_dir = Path("labels/val")
        
        if val_images_dir.exists() and val_labels_dir.exists():
            val_images = len(list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")))
            val_labels = len(list(val_labels_dir.glob("*.txt")))
            print(f"📸 Validation Images: {val_images}")
            print(f"🏷️  Validation Labels: {val_labels}")
            
            if val_images == 0:
                print("⚠️  No validation images found!")
                return False
        else:
            print("⚠️  Validation directories not found!")
            return False
        
        # PyTorch info
        print(f"🔧 PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Using CPU")
            
        return True
    
    def load_model(self):
        """Load the trained model"""
        try:
            print(f"\n🤖 Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Model information
            if hasattr(self.model.model, 'parameters'):
                total_params = sum(p.numel() for p in self.model.model.parameters())
                print(f"📊 Parameters: {total_params:,}")
            
            print("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def run_validation(self):
        """Run comprehensive validation"""
        if not self.model:
            print("❌ Model not loaded!")
            return None
            
        print("\n🔍 Running Model Validation...")
        print("=" * 40)
        print("📊 This will evaluate model performance on validation set")
        print("⏱️  Please wait for validation to complete...")
        print("-" * 40)
        
        try:
            # Run validation
            self.validation_results = self.model.val(
                data=self.data_config,
                save_json=True,
                save_hybrid=True,
                plots=True,
                verbose=True,
                split='val'
            )
            
            print("\n✅ Validation completed successfully!")
            return self.validation_results
            
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            return None
    
    def analyze_results(self):
        """Analyze and display validation results"""
        if not self.validation_results:
            print("❌ No validation results to analyze!")
            return
            
        print("\n📊 VALIDATION RESULTS ANALYSIS")
        print("=" * 50)
        
        # Extract key metrics
        results = self.validation_results
        
        # Overall metrics
        print("🎯 Overall Performance:")
        print(f"   Precision (P): {results.box.mp:.4f}")
        print(f"   Recall (R): {results.box.mr:.4f}")
        print(f"   mAP@0.5: {results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
        
        # Per-class metrics if available
        if hasattr(results.box, 'ap_class_index') and len(results.box.ap_class_index) > 0:
            print("\n📈 Per-Class Performance:")
            class_names = self.model.names
            
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = class_names.get(class_idx, f"Class {class_idx}")
                precision = results.box.p[i] if i < len(results.box.p) else 0
                recall = results.box.r[i] if i < len(results.box.r) else 0
                ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
                ap = results.box.ap[i] if i < len(results.box.ap) else 0
                
                priority = "🔴 HIGH" if class_name.lower() == 'human' else "🟡 MED"
                
                print(f"   {priority} {class_name}:")
                print(f"      Precision: {precision:.4f}")
                print(f"      Recall: {recall:.4f}")
                print(f"      AP@0.5: {ap50:.4f}")
                print(f"      AP@0.5:0.95: {ap:.4f}")
        
        # Speed metrics
        if hasattr(results, 'speed'):
            speed = results.speed
            print(f"\n⚡ Performance Speed:")
            print(f"   Preprocess: {speed.get('preprocess', 0):.1f}ms")
            print(f"   Inference: {speed.get('inference', 0):.1f}ms")
            print(f"   Postprocess: {speed.get('postprocess', 0):.1f}ms")
            print(f"   Total: {sum(speed.values()):.1f}ms per image")
        
        # Model file info
        model_size = os.path.getsize(self.model_path) / (1024*1024)
        print(f"\n💾 Model Information:")
        print(f"   File Size: {model_size:.1f} MB")
        print(f"   Classes: {list(self.model.names.values())}")
        
        return results
    
    def generate_detailed_report(self):
        """Generate detailed validation report"""
        if not self.validation_results:
            print("❌ No results to report!")
            return
            
        print("\n📋 DETAILED VALIDATION REPORT")
        print("=" * 50)
        
        results = self.validation_results
        
        # Create comprehensive report
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "data_config": self.data_config,
            "overall_metrics": {
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
                "mAP50": float(results.box.map50),
                "mAP50_95": float(results.box.map),
                "fitness": float(results.fitness) if hasattr(results, 'fitness') else 0
            },
            "class_metrics": {},
            "model_info": {
                "classes": list(self.model.names.values()),
                "num_classes": len(self.model.names),
                "model_size_mb": os.path.getsize(self.model_path) / (1024*1024)
            }
        }
        
        # Per-class metrics
        if hasattr(results.box, 'ap_class_index'):
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = self.model.names.get(class_idx, f"class_{class_idx}")
                report["class_metrics"][class_name] = {
                    "precision": float(results.box.p[i]) if i < len(results.box.p) else 0,
                    "recall": float(results.box.r[i]) if i < len(results.box.r) else 0,
                    "ap50": float(results.box.ap50[i]) if i < len(results.box.ap50) else 0,
                    "ap50_95": float(results.box.ap[i]) if i < len(results.box.ap) else 0
                }
        
        # Speed metrics
        if hasattr(results, 'speed'):
            report["speed_metrics"] = dict(results.speed)
        
        # Save report
        report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"📄 Detailed report saved: {report_file}")
        
        # Performance assessment
        self.assess_model_performance(report)
        
        return report
    
    def assess_model_performance(self, report):
        """Assess model performance and provide recommendations"""
        print("\n🎯 PERFORMANCE ASSESSMENT")
        print("=" * 40)
        
        overall = report["overall_metrics"]
        
        # Overall assessment
        map50 = overall["mAP50"]
        map50_95 = overall["mAP50_95"]
        
        if map50 >= 0.7:
            performance = "🟢 EXCELLENT"
        elif map50 >= 0.5:
            performance = "🟡 GOOD"
        elif map50 >= 0.3:
            performance = "🟠 FAIR"
        else:
            performance = "🔴 NEEDS IMPROVEMENT"
            
        print(f"Overall Performance: {performance}")
        print(f"mAP@0.5: {map50:.1%}")
        
        # Specific assessments for disaster management
        print("\n🚨 Disaster Management Assessment:")
        
        class_metrics = report["class_metrics"]
        
        # Human detection assessment (critical for rescue)
        if "human" in class_metrics:
            human_recall = class_metrics["human"]["recall"]
            human_precision = class_metrics["human"]["precision"]
            
            print(f"👤 Human Detection:")
            print(f"   Recall: {human_recall:.1%} ({'✅ Good' if human_recall >= 0.8 else '⚠️ Needs improvement'})")
            print(f"   Precision: {human_precision:.1%}")
            
            if human_recall < 0.8:
                print("   🔴 CRITICAL: Low recall for human detection may miss people in need!")
        
        # Debris detection assessment
        if "debris" in class_metrics:
            debris_metrics = class_metrics["debris"]
            print(f"🧱 Debris Detection:")
            print(f"   Recall: {debris_metrics['recall']:.1%}")
            print(f"   Precision: {debris_metrics['precision']:.1%}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        
        if map50 < 0.5:
            print("🔄 Model needs more training:")
            print("   - Increase training epochs (50-100)")
            print("   - Add more diverse training data")
            print("   - Consider data augmentation")
            print("   - Try larger YOLOv8 variant (s, m, l, x)")
        
        if overall["precision"] < 0.5:
            print("🎯 Improve precision:")
            print("   - Increase confidence threshold")
            print("   - Review and clean training labels")
            print("   - Balance class distribution")
        
        if overall["recall"] < 0.8:
            print("📈 Improve recall:")
            print("   - Lower confidence threshold")
            print("   - Add more positive examples")
            print("   - Review missed detections")
        
        print("\n✅ Model is ready for integration into Vihangam system!")
    
    def save_validation_plots(self):
        """Save validation plots if available"""
        if not self.validation_results:
            return
            
        # Check for generated plots
        save_dir = Path(self.validation_results.save_dir) if hasattr(self.validation_results, 'save_dir') else None
        
        if save_dir and save_dir.exists():
            plots = list(save_dir.glob("*.png"))
            if plots:
                print(f"\n📊 Validation plots saved in: {save_dir}")
                for plot in plots[:5]:  # Show first 5 plots
                    print(f"   📈 {plot.name}")
        
        return save_dir

def main():
    """Main validation function"""
    print("🚁 Starting YOLOv8 Model Validation for Vihangam")
    
    # Initialize validator
    try:
        validator = VihangamModelValidator()
    except FileNotFoundError as e:
        print(e)
        print("\n💡 Available models:")
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            for model_path in runs_dir.rglob("best.pt"):
                print(f"   📦 {model_path}")
        return
    
    # Validate environment
    if not validator.validate_environment():
        print("❌ Environment validation failed!")
        return
    
    # Load model
    if not validator.load_model():
        print("❌ Model loading failed!")
        return
    
    # Run validation
    print(f"\n🚀 Starting validation process...")
    results = validator.run_validation()
    
    if results:
        # Analyze results
        validator.analyze_results()
        
        # Generate detailed report
        validator.generate_detailed_report()
        
        # Save plots
        validator.save_validation_plots()
        
        print("\n" + "=" * 60)
        print("🎉 VALIDATION COMPLETE!")
        print("=" * 60)
        print("✅ Model performance evaluated")
        print("📊 Detailed metrics generated")
        print("📈 Validation plots saved")
        print("📄 JSON report created")
        print("\n💡 Your model is ready for deployment in the Vihangam system!")
        print("🚁 Integrate with your disaster management dashboard")
        print("=" * 60)
        
    else:
        print("❌ Validation failed!")

if __name__ == "__main__":
    main()