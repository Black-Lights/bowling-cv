# config/config.py

class Config:
    # =========================================================================
    # SYSTEM CONFIGURATION
    # =========================================================================
    DEBUG_MODE = True  # Enable visualization windows
    WRITE_VIDEO = False
    
    # =========================================================================
    # LANE MASKING (Phase 1 Integration)
    # =========================================================================
    # Pixels to expand the lane mask to ensure we catch edge balls
    LANE_MASK_MARGIN = 20 

    # =========================================================================
    # BACKGROUND SUBTRACTION (MOG2)
    # =========================================================================
    # History: Length of the history. Higher = adapts slower to light changes.
    BG_HISTORY = 500
    # VarThreshold: Mahalanobis distance squared. Lower = more sensitive.
    BG_VAR_THRESHOLD = 16
    # DetectShadows: Essential for bowling. Marks shadows as gray (127).
    BG_DETECT_SHADOWS = True
    
    # =========================================================================
    # MORPHOLOGICAL OPERATIONS
    # =========================================================================
    # Kernel for removing noise (Opening)
    MORPH_OPEN_KERNEL_SIZE = (3, 3)
    # Kernel for filling holes in the ball (Closing)
    MORPH_CLOSE_KERNEL_SIZE = (5, 5)
    
    # =========================================================================
    # GEOMETRIC FILTERS (The Bowler Rejection Logic)
    # =========================================================================
    # Area: Min/Max pixels. 
    # NOTE: These should ideally be dynamic based on Y-position.
    BALL_MIN_AREA = 100
    BALL_MAX_AREA = 2500
    
    # Circularity: 1.0 is perfect circle. 0.7 allows for motion blur.
    # Bowlers (human shapes) typically score < 0.5
    MIN_CIRCULARITY = 0.65
    
    # Aspect Ratio: Width / Height. Ball is roughly 1.0.
    MIN_ASPECT_RATIO = 0.6
    MAX_ASPECT_RATIO = 1.4
    
    # =========================================================================
    # TRACKING & KALMAN FILTER
    # =========================================================================
    # State Machine: Frames to wait before confirming a track
    ACQUISITION_BUFFER = 3
    # State Machine: Frames to wait before declaring 'LOST'
    MAX_MISSED_FRAMES = 5
    
    # ROI Buffer: Size of the search window around predicted position (pixels)
    TRACKING_ROI_BUFFER = 100
    
    # Kalman Covariances
    # Process Noise (Q): Trust in the constant velocity model.
    # Higher = we expect more curvature/acceleration.
    KF_PROCESS_NOISE = 0.03
    
    # Measurement Noise (R): Trust in the detection accuracy.
    # Higher = detection is noisy, rely more on prediction (smoother).
    KF_MEASUREMENT_NOISE = 0.5