import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def calculate_mouth_openness(landmarks, upper_lip_indices, lower_lip_indices):
    """Calculate the average vertical distance between upper lip and lower lip."""
    upper_lip = np.mean([landmarks[i][1] for i in upper_lip_indices])
    lower_lip = np.mean([landmarks[i][1] for i in lower_lip_indices])
    return abs(lower_lip - upper_lip)

def crop_mouth_with_talking_detection(video_path, output_path, openness_threshold=12, frame_check_count=5, padding=100):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = 88  # Desired width of the cropped mouth region
    frame_height = 88  # Desired height of the cropped mouth region
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is a common codec for mp4 files
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    fixed_bbox = None  # To store the fixed bounding box for cropping
    outer_lip_indices = [61, 291, 146, 375, 78, 308, 95, 324]
    # upper_lip_indices = [13, 14, 15]  # Upper lip landmarks
    upper_lip_indices = [11, 12, 13]  # Upper lip landmarks
    # lower_lip_indices = [17, 18, 19]  # Lower lip landmarks
    lower_lip_indices = [14, 15, 16]  # Lower lip landmarks

    previous_frames = []  # List to hold the last five frames for motion detection
    mouth_openness_list = []  # Track mouth openness to detect talking
    talking_frames = []   # List to keep track of talking frames (by indices)
    current_talking = False  # Boolean to track if the person is currently talking
    frame_idx = 0  # To track the frame index

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for face landmark detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the landmark positions for the outer and inner lips
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]

                # Calculate the bounding box for the mouth region on the first valid frame
                if fixed_bbox is None:
                    mouth_points = [landmarks[i] for i in outer_lip_indices]
                    x_min = min(mouth_points, key=lambda p: p[0])[0]
                    y_min = min(mouth_points, key=lambda p: p[1])[1]
                    x_max = max(mouth_points, key=lambda p: p[0])[0]
                    y_max = max(mouth_points, key=lambda p: p[1])[1]

                    # Add padding and ensure coordinates are within bounds
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame.shape[1], x_max + padding)
                    y_max = min(frame.shape[0], y_max + padding)

                    # Store the bounding box (x_min, y_min, x_max, y_max) for future frames
                    fixed_bbox = (x_min, y_min, x_max, y_max)

                # Apply the fixed bounding box for cropping
                x_min, y_min, x_max, y_max = fixed_bbox
                mouth_region = frame[y_min:y_max, x_min:x_max]

                if mouth_region.size == 0:
                    continue

                # Resize the cropped mouth region
                resized_mouth = cv2.resize(mouth_region, (frame_width, frame_height))

                # Convert the resized mouth region to grayscale for further processing
                gray_mouth = cv2.cvtColor(resized_mouth, cv2.COLOR_BGR2GRAY)

                # Calculate the mouth openness using landmarks
                mouth_openness = calculate_mouth_openness(landmarks, upper_lip_indices, lower_lip_indices)
                mouth_openness_list.append(mouth_openness)

                # Keep track of the last five frames
                if len(mouth_openness_list) >= frame_check_count:
                    # Check if talking based on mouth openness changes
                    if np.mean(mouth_openness_list[-frame_check_count:]) > openness_threshold:
                        # Person is talking, so mark these frames as talking
                        if not current_talking:
                            # Add the previous six frames (if available)
                            start_idx = max(0, frame_idx - 10)
                            talking_frames.extend(range(start_idx, frame_idx + 1))
                        current_talking = True
                        talking_frames.append(frame_idx)
                    else:
                        if current_talking:
                            # Add the current and next six frames
                            talking_frames.extend(range(frame_idx, frame_idx + 1 + 10))
                        current_talking = False

                # Remove old entries to maintain the sliding window for checking mouth openness
                if len(mouth_openness_list) > frame_check_count:
                    mouth_openness_list.pop(0)

                # Save only the talking frames (by frame index)
                if True:#frame_idx in talking_frames:
                    out.write(resized_mouth)

        frame_idx += 1
    print(talking_frames.__len__())
    # Release video resources
    cap.release()
    out.release()

    print(f'Video saved to {output_path}')
