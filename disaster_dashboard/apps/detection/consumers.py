import json
import asyncio
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .yolo_handler import get_yolo_detector
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class DetectionConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.room_name = 'detection'
        self.room_group_name = f'detection_{self.room_name}'
        self.detection_active = False
        self.confidence_threshold = 0.5
        self.detection_task = None
    
    async def connect(self):
        """Accept WebSocket connection and join detection group"""
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Send initial status
        await self.send(text_data=json.dumps({
            'type': 'connection_status',
            'status': 'connected',
            'message': 'Connected to Vihangam Detection System',
            'timestamp': datetime.now().isoformat()
        }))
        
        logger.info(f"WebSocket connected: {self.channel_name}")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        # Stop detection if active
        if self.detection_active:
            await self.stop_detection_internal()
        
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        
        logger.info(f"WebSocket disconnected: {self.channel_name}, code: {close_code}")
    
    async def receive(self, text_data):
        """Handle messages from WebSocket"""
        try:
            data = json.loads(text_data)
            command = data.get('command')
            
            if command == 'start_detection':
                await self.start_detection(data)
            elif command == 'stop_detection':
                await self.stop_detection_external()
            elif command == 'update_settings':
                await self.update_detection_settings(data)
            elif command == 'get_model_info':
                await self.send_model_info()
            elif command == 'ping':
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }))
            else:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Unknown command: {command}'
                }))
                
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def start_detection(self, data):
        """Start the detection process"""
        try:
            if self.detection_active:
                await self.send(text_data=json.dumps({
                    'type': 'warning',
                    'message': 'Detection already active'
                }))
                return
            
            # Update settings
            self.confidence_threshold = data.get('confidence', 0.5)
            model_path = data.get('model_path')
            
            # Initialize YOLO detector
            detector = await self.get_detector(model_path)
            if not detector:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'Failed to initialize YOLO detector'
                }))
                return
            
            # Start detection
            self.detection_active = True
            self.detection_task = asyncio.create_task(self.detection_loop())
            
            await self.send(text_data=json.dumps({
                'type': 'detection_started',
                'message': f'Detection started with confidence {self.confidence_threshold}',
                'confidence': self.confidence_threshold,
                'model_info': await self.get_model_info_sync(),
                'timestamp': datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(f"Start detection error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def stop_detection_external(self):
        """Stop detection (called from external command)"""
        await self.stop_detection_internal()
        await self.send(text_data=json.dumps({
            'type': 'detection_stopped',
            'message': 'Detection stopped by user',
            'timestamp': datetime.now().isoformat()
        }))
    
    async def stop_detection_internal(self):
        """Internal method to stop detection"""
        self.detection_active = False
        if self.detection_task and not self.detection_task.done():
            self.detection_task.cancel()
            try:
                await self.detection_task
            except asyncio.CancelledError:
                pass
    
    async def detection_loop(self):
        """Main detection loop - simulates continuous detection"""
        frame_count = 0
        try:
            while self.detection_active:
                frame_count += 1
                
                # Simulate detection results (in real implementation, process actual video frames)
                detection_data = await self.generate_mock_detection(frame_count)
                
                # Send detection update
                await self.send(text_data=json.dumps({
                    'type': 'detection_update',
                    'data': detection_data
                }))
                
                # Wait before next frame (simulate FPS)
                await asyncio.sleep(1/30)  # 30 FPS
                
        except asyncio.CancelledError:
            logger.info("Detection loop cancelled")
        except Exception as e:
            logger.error(f"Detection loop error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Detection error: {str(e)}'
            }))
            self.detection_active = False
    
    async def generate_mock_detection(self, frame_count):
        """Generate mock detection data (replace with real detection in production)"""
        # Simulate varying detection results
        num_objects = random.randint(0, 4)
        detections = []
        
        disaster_classes = ['person', 'car', 'truck', 'motorbike', 'bus', 'debris']
        
        for i in range(num_objects):
            class_name = random.choice(disaster_classes)
            is_high_priority = class_name in ['person', 'debris']
            
            detection = {
                'id': f'obj_{frame_count}_{i}',
                'bbox': [
                    random.randint(50, 300),
                    random.randint(50, 200), 
                    random.randint(80, 150),
                    random.randint(60, 120)
                ],
                'confidence': round(random.uniform(0.6, 0.95), 3),
                'class_name': class_name,
                'is_disaster_related': True,
                'is_high_priority': is_high_priority,
                'area': random.randint(2000, 8000)
            }
            detections.append(detection)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'frame_id': frame_count,
            'detections': detections,
            'total_count': len(detections),
            'disaster_count': len(detections),  # All are disaster-related in simulation
            'high_priority_count': sum(1 for d in detections if d['is_high_priority']),
            'fps': random.randint(28, 32),
            'processing_time': round(random.uniform(0.02, 0.05), 3),
            'average_confidence': round(sum(d['confidence'] for d in detections) / len(detections), 3) if detections else 0
        }
    
    async def update_detection_settings(self, data):
        """Update detection settings"""
        try:
            settings = data.get('settings', {})
            
            if 'confidence' in settings:
                self.confidence_threshold = float(settings['confidence'])
            
            await self.send(text_data=json.dumps({
                'type': 'settings_updated',
                'settings': {
                    'confidence': self.confidence_threshold
                },
                'message': 'Detection settings updated',
                'timestamp': datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(f"Settings update error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def send_model_info(self):
        """Send current model information"""
        try:
            model_info = await self.get_model_info_sync()
            await self.send(text_data=json.dumps({
                'type': 'model_info',
                'model_info': model_info,
                'timestamp': datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(f"Model info error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    @database_sync_to_async
    def get_detector(self, model_path=None):
        """Get YOLO detector instance"""
        try:
            return get_yolo_detector(model_path)
        except Exception as e:
            logger.error(f"Failed to get detector: {e}")
            return None
    
    @database_sync_to_async
    def get_model_info_sync(self):
        """Get model info synchronously"""
        try:
            detector = get_yolo_detector()
            return detector.get_model_info()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
    
    # Group message handlers
    async def detection_broadcast(self, event):
        """Handle broadcast messages to the detection group"""
        await self.send(text_data=json.dumps({
            'type': 'broadcast',
            'message': event['message'],
            'data': event.get('data', {})
        }))
