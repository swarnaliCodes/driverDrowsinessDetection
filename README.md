Driver Drowsiness Detection using Deep Learning (YOLO-Based)

1. Overview

    Driver fatigue is one of the leading causes of road accidents worldwide. This project presents a real-time, vision-based Driver Drowsiness Detection System       built using deep learning-based object detection techniques.
    
    The system leverages YOLO (You Only Look Once) for detecting:
    
    Eye closure
    
    Yawning behavior
    
    Drowsiness state
    
    To improve robustness in real-world conditions, a temporal smoothing mechanism is implemented to reduce false positives and stabilize alert generation. The       system is optimized for real-time inference and can operate on CPU-based systems without requiring a GPU.

2. Problem Statement

    Traditional fatigue detection systems often rely on handcrafted features or single-frame analysis, which can lead to unstable predictions and frequent false      alarms. This project addresses these limitations by:
    
    Using deep learning-based object detection for improved accuracy
    
    Modeling temporal consistency across consecutive frames
    
    Designing a threshold-based alerting mechanism for stable real-time performance

3. Technical Approach
   
    3.1 Data Preparation
    
        Annotated dataset containing:
            
            Eyeclosed
            
            Yawn
        
        Train/validation split for performance evaluation
        
        Data augmentation to improve generalization
  
    3.2 Model Training
    
        Transfer learning using pretrained YOLO weights
        
        Optimizer: AdamW
        
        Cosine learning rate scheduling
        
        Early stopping with defined patience
  
    3.3 Temporal Smoothing Logic
    
        Instead of triggering alerts on single-frame detections, the system:
        
        Tracks detections across consecutive frames
        
        Applies persistence thresholds
        
        Generates an alert only when drowsiness indicators are sustained
        
        This approach significantly improves real-world reliability.

4. Model Performance (Validation Results)
   
        Metric	Score
        mAP@50	0.9254
        mAP@50–95	0.4995
        Precision	0.9151
        Recall	0.8865
  
     These results indicate strong detection capability with balanced precision and recall, demonstrating suitability for safety-critical applications.

5. Tech Stack

        Python
        
        PyTorch
        
        OpenCV
        
        YOLO
        
        NumPy

6. Research Perspective

      This project focuses on improving the reliability of vision-based fatigue detection systems in unconstrained environments. Unlike conventional frame-level        approaches, the proposed architecture integrates:
      
        Deep learning-based object detection
        
        Temporal behavioral modeling
        
        Threshold-based alert mechanisms
    
      The system achieves high detection accuracy while maintaining computational efficiency suitable for real-time deployment. It provides a strong foundation         for further research in:
      
        Edge deployment in automotive systems
        
        Embedded AI for intelligent vehicles
        
        Multi-modal fatigue detection frameworks
  
7. Skills Demonstrated

      This project demonstrates:
      
        Deep learning model training and optimization
        
        Real-time computer vision system design
        
        Object detection and evaluation metrics
        
        Performance tuning and validation
        
        Deployment-oriented ML pipeline development
        
        Applied problem-solving for safety-critical systems
