# app/tracker.py
from ultralytics import YOLO
import torch

class VehicleTracker:
    def __init__(self, model_path='yolo11n.pt'):
        # Check if GPU is available, otherwise fallback to CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = YOLO(model_path)
        # Move model to the chosen device
        self.model.to(self.device)

    def track(self, frame):
        # Pass the device to the track method
        results = self.model.track(
            source=frame, 
            persist=True, 
            tracker="bytetrack.yaml",
            conf=0.3,
            iou=0.5,
            classes=[2, 3, 5, 7], # car, motorcycle, bus, truck [cite: 14, 41]
            device=self.device    # This forces the use of the GPU
        )
        return results[0]