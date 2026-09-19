\# BasketTube AI Game Analyzer: Project Report



\# Written Report



\## 1. Project Overview



This project implements a basketball game analysis system that combines commentary-grounded question answering with video-grounded action verification. Basketball video is difficult to analyze because multiple players move simultaneously, the ball is small and fast, broadcast camera angles change, and many actions require temporal context. To make the problem manageable, the project separates the work into two tasks.



Task 1 analyzes player performance using spoken commentary from the full game video. The commentary is transcribed with Whisper, chunked into timestamped text windows, and retrieved using TF-IDF similarity for user questions.



Task 2 verifies player actions from video clips. The original Roboflow basketball detector was attempted, but local ONNX inference failed in Colab due to CUDA/CPU device-binding errors. Therefore, the final reproducible pipeline uses YOLOv8 as a fallback object detector. YOLO is used to detect visible people and plausible sports-ball locations, generate representative evidence frames, and support an action table.



\## 2. Task 1 Method



The full game audio was extracted using FFmpeg and transcribed using Whisper through the `faster-whisper` library. The transcript was converted into timestamped chunks. For each user query, TF-IDF retrieval selected the most relevant transcript chunks. The answer generator then produced a direct answer, listed retrieved timestamped evidence, separated facts from interpretation, and included a limitation note.



This approach keeps Task 1 grounded in commentary rather than unsupported assumptions.



\## 3. Task 1 Results



The system was tested with three questions:



1\. Who had the strongest scoring impact?

2\. How did the offense create chances?

3\. Which moments mention defensive pressure or stops?



The retrieved commentary evidence highlighted Stephen Curry’s scoring and offensive value, offensive creation through self-creation, screens, extra passing, and defensive moments involving Wiggins, Green, and team defensive coverage.



\## 4. Task 2 Method



The source directory contains 10 short basketball clips. Each clip is treated as a candidate play segment. For each segment, a representative frame is extracted and processed with YOLOv8. The detector identifies visible players as `person` detections and identifies the basketball as `sports ball` when possible.



The final action table includes timestamp, player description, identity confidence, action category, detection counts, visual evidence explanation, and representative frame path.



To strengthen visual verification beyond a single representative frame, the notebook also samples multiple frames from one candidate play window. This frame-sequence evidence provides limited temporal context for player and ball detections. It is not full object tracking, but it improves the action evidence by showing how detections appear across several moments in the clip.



\## 5. Task 2 Results



The system includes frame-sequence evidence, but it does not implement full multi-object tracking across the entire clip. The actions include offensive spacing, passing or ball-movement context, defensive pressure, dribbling or drive setup, and ball possession context.



Because player names and jersey numbers are not reliably readable, the system avoids unsupported player-name claims and instead uses conservative labels such as “unknown offensive player” or “unknown defender.” This keeps the analysis honest and grounded in visible evidence.



\## 6. Commentary vs Video Comparison



Task 1 provides high-level context from spoken commentary, while Task 2 provides frame-level visual grounding. Commentary can identify player impact and explain strategy, but it may be subjective or contain ASR errors. Video evidence can show visible spacing, defenders, ball location, and action windows, but it may not reliably identify players or detect the ball in every frame.



Together, the two tasks form a complementary multimodal analysis system.



\## 7. Limitations



The main limitations are:



\- Whisper may misrecognize player names or phrases.

\- YOLOv8 is a general-purpose detector, not a basketball-specific action classifier.

\- The ball is small, fast, and sometimes occluded.

\- Broadcast camera angles make jersey numbers difficult to read.

\- The system does not fully track players across time.

\- The action labels are evidence-supported interpretations, not perfect ground-truth annotations.

\- Some actions such as assists, screens, and defensive pressure require longer temporal context than a single representative frame.

\- The system includes frame-sequence evidence, but it does not implement full multi-object tracking across the entire clip.

\## 8. Future Improvements



With more time, the system could be improved by adding a basketball-specific detector, robust ball tracking, jersey OCR, pose estimation, court homography, player tracking, and a video-language model for direct clip-level action recognition. A stronger version could also align commentary timestamps with visual clips more precisely and compare commentator claims against visual evidence.

