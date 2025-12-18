from ultralytics import YOLO

class VehicleTracker:
    def __init__(self, model_path='yolo11n.pt'):
        self.model = YOLO(model_path)

    def track(self, frame):
        # persist=True is key for keeping IDs across frames [cite: 79]
        results = self.model.track(
            source=frame, 
            persist=True, 
            tracker="bytetrack.yaml",
            conf=0.3, # Confidence threshold
            iou=0.5,  # Intersection over Union threshold
            classes=[2, 3, 5, 7] # COCO classes: car, motorcycle, bus, truck [cite: 14, 41]
        )
        return results[0]