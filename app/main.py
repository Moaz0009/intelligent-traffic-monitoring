import cv2
from tracker import VehicleTracker
from google.colab.patches import cv2_imshow

# Paths from your setup
VIDEO_PATH = "/content/drive/MyDrive/intelligent-traffic-monitoring/Data/Vehicle Dataset Sample 3.mp4"

def main():
    tracker = VehicleTracker()
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Get tracking results
        tracked_frame_data = tracker.track(frame)
        
        # Visualize the tracking (boxes + IDs)
        annotated_frame = tracked_frame_data.plot()

        # Display (Only every 5th frame to save Colab memory if needed)
        cv2_imshow(annotated_frame)
        
        # Break for testing (Phase 2 confirmation)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()