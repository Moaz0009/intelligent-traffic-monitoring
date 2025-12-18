import cv2
import os
import numpy as np

class RestrictedAreaMonitor:
    def __init__(self, polygon_coords, snapshot_dir="snapshots"):
        self.polygon = np.array(polygon_coords, np.int32).reshape((-1, 1, 2))
        self.snapshot_dir = snapshot_dir
        # Using a set to ensure only one snapshot per unique tracking ID 
        self.violated_ids = set() 
        
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

    def check_violations(self, frame, results):
        if results.boxes.id is None:
            return

        ids = results.boxes.id.cpu().numpy().astype(int)
        bboxes = results.boxes.xyxy.cpu().numpy()

        for track_id, bbox in zip(ids, bboxes):
            # Calculate center point for detection [cite: 34]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)

            # Check if center enters the polygon [cite: 35]
            is_inside = cv2.pointPolygonTest(self.polygon, (cx, cy), False)
            
            if is_inside >= 0: 
                if track_id not in self.violated_ids:
                    self.violated_ids.add(track_id)
                    
                    # Create a copy to draw on for the evidence snapshot 
                    snapshot_frame = frame.copy()
                    
                    # Draw a RED rectangle around the violating vehicle
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Add a label for the snapshot
                    cv2.putText(snapshot_frame, f"VIOLATION ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # Save the evidence [cite: 24, 38]
                    snapshot_path = os.path.join(self.snapshot_dir, f"violation_{track_id}.jpg")
                    cv2.imwrite(snapshot_path, snapshot_frame)
                    print(f"Captured evidence for Vehicle {track_id}")