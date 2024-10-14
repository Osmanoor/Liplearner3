import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def crop_mouth_fixed(video_path, output_path, padding=100):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = 88  # Desired width of the cropped mouth region
    frame_height = 88  # Desired height of the cropped mouth region

    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    fixed_bbox = None  # To store the fixed bounding box for cropping
    outer_lip_indices = [61, 291, 146, 375, 78, 308, 95, 324]  # Lip landmarks

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for face landmark detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the landmark positions
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]

                # Calculate the bounding box for the mouth region on the first valid frame
                if fixed_bbox is None:
                    mouth_points = [landmarks[i] for i in outer_lip_indices]
                    x_min = min(mouth_points, key=lambda p: p[0])[0]
                    y_min = min(mouth_points, key=lambda p: p[1])[1]
                    x_max = max(mouth_points, key=lambda p: p[0])[0]
                    y_max = max(mouth_points, key=lambda p: p[1])[1]

                    # Add padding and ensure coordinates are within frame bounds
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

                # Save the resized mouth region to the output video
                out.write(resized_mouth)

    # Release video resources
    cap.release()
    out.release()

    print(f'Video saved to {output_path}')
