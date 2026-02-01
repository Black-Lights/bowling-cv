import cv2
import numpy as np
from config.config import Config
from core.ball_detection import BallDetector
from core.kalman_filter import BowlingKalmanFilter
# Assume detection_utils and master_line_computation are available from Phase 1

def main():
    # --- SETUP ---
    cap = cv2.VideoCapture('bowling_sample.mp4')
    
    # Phase 1: Get Lane Geometry (Mocking the output here)
    # In production: lane_data = master_line_computation.process(first_frame)
    lane_polygon = np.array([, , , ]) # Example
    
    detector = BallDetector()
    tracker = BowlingKalmanFilter()
    
    # State Variables
    state = "ACQUIRE" # ACQUIRE, TRACK, LOST
    trajectory =
    missed_frames = 0
    acquisition_counter = 0
    
    # Static Lane Mask (Optimized: Created once)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    lane_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lane_mask, [lane_polygon], 255)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        debug_frame = frame.copy()
        
        # --- STATE MACHINE ---
        
        if state == "ACQUIRE":
            # Global Search
            candidate, mask = detector.detect(frame, lane_mask)
            
            if candidate:
                cx, cy, cnt = candidate
                acquisition_counter += 1
                
                # Draw Candidate
                cv2.circle(debug_frame, (cx, cy), 10, (0, 255, 255), 2)
                
                if acquisition_counter >= Config.ACQUISITION_BUFFER:
                    # Transition -> TRACK
                    state = "TRACK"
                    tracker.initialize(cx, cy)
                    trajectory = [(cx, cy)]
                    missed_frames = 0
                    print("Ball Acquired. Switching to TRACK mode.")
            else:
                acquisition_counter = 0

        elif state == "TRACK":
            # Prediction Step
            pred_x, pred_y = tracker.predict()
            
            # Define ROI
            r = Config.TRACKING_ROI_BUFFER
            x1, y1 = max(0, pred_x - r), max(0, pred_y - r)
            x2, y2 = min(w, pred_x + r), min(h, pred_y + r)
            
            roi_frame = frame[y1:y2, x1:x2]
            roi_mask = lane_mask[y1:y2, x1:x2]
            
            # Local Search in ROI
            candidate, roi_fg_mask = detector.detect(roi_frame, roi_mask, offset=(x1, y1))
            
            # Update Debug Mask (Paste ROI mask into global view)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = roi_fg_mask
            
            if candidate:
                cx, cy, cnt = candidate
                # Correction Step
                est_x, est_y = tracker.correct(cx, cy)
                trajectory.append((est_x, est_y))
                missed_frames = 0
                
                # Visualization
                cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1) # Raw
                cv2.circle(debug_frame, (est_x, est_y), 5, (0, 255, 0), -1) # Kalman
            else:
                # No detection: Trust Prediction
                trajectory.append((pred_x, pred_y))
                missed_frames += 1
                cv2.circle(debug_frame, (pred_x, pred_y), 5, (255, 0, 0), -1) # Pred
                
                if missed_frames > Config.MAX_MISSED_FRAMES:
                    state = "ACQUIRE"
                    missed_frames = 0
                    acquisition_counter = 0
                    print("Track Lost. Reverting to Global Search.")

            # Draw ROI Box
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # --- VISUALIZATION ---
        
        # Draw Trajectory
        if len(trajectory) > 1:
            pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_frame, [pts], False, (0, 255, 0), 2)
            
        # Draw Lane Boundaries (Phase 1)
        cv2.polylines(debug_frame, [lane_polygon], True, (255, 0, 0), 1)

        if Config.DEBUG_MODE:
            cv2.imshow("Tracking", debug_frame)
            cv2.imshow("Mask", mask)
            
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()