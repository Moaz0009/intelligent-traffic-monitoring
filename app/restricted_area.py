import cv2
import os
import numpy as np

class RestrictedAreaMonitor:
    def __init__(self, polygon_coords, snapshot_dir="snapshots"):
        # Convert list of coordinates to a numpy array for OpenCV logic [cite: 32, 80]
        self.polygon = np.array(polygon_coords, np.int32).reshape((-1, 1, 2))
        self.snapshot_dir = snapshot_dir
        self.violated_ids = set() # Prevent repeated alerts for the same ID [cite: 36]
        
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

    def check_violations(self, frame, results):
        # Ensure we have tracking IDs before processing [cite: 15]
        if results.boxes.id is None:
            return

        # Get tracking IDs and bounding boxes
        ids = results.boxes.id.cpu().numpy().astype(int)
        bboxes = results.boxes.xyxy.cpu().numpy()

        for track_id, bbox in zip(ids, bboxes):
            # Calculate the center point of the vehicle's bounding box [cite: 34]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)

            # Check if the center point is inside the restricted polygon [cite: 34, 35]
            is_inside = cv2.pointPolygonTest(self.polygon, (cx, cy), False)
            
            if is_inside >= 0: # 0 = on edge, 1 = inside
                if track_id not in self.violated_ids:
                    self.violated_ids.add(track_id)
                    
                    # Capture current frame as evidence for the violation [cite: 24, 37]
                    snapshot_path = os.path.join(self.snapshot_dir, f"violation_id_{track_id}.jpg")
                    cv2.imwrite(snapshot_path, frame)
                    print(f"VIOLATION: Vehicle {track_id} entered restricted area. Snapshot saved.")