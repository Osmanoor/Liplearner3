import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
import torch
import torch.nn.functional as F


# Define the path to your 'AnasLips' folder
base_path = '/content/drive/MyDrive/AnasLips'

# Function to load video
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Augmentation functions (as defined in the previous response)
def TensorRandomFlip(tensor):
    if random.random() > 0.5:
        return torch.flip(tensor, dims=[4])
    return tensor

def TensorRandomCrop(tensor, size):
    _, _, t, h, w = tensor.size()
    th, tw = size

    # Ensure crop size is not larger than the input size
    th = min(th, h)
    tw = min(tw, w)

    if w == tw and h == th:
        return tensor

    x1 = random.randint(0, w - tw) if w > tw else 0
    y1 = random.randint(0, h - th) if h > th else 0

    return tensor[:,:,:,y1:y1+th,x1:x1+tw]

# Update CenterCrop function to handle smaller videos
def CenterCrop(batch_img, size):
    _, h, w, _ = batch_img.shape
    th, tw = size
    # Ensure crop size is not larger than the input size
    th = min(th, h)
    tw = min(tw, w)
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return batch_img[:, y1:y1+th, x1:x1+tw]

def RandomFrameDrop(batch_img, duration):
    remaining_list = range(len(batch_img))
    if random.random() > 0.5:
        drop_margin = int((len(batch_img) - duration.sum() * 0.8) / 2)
        drop_start = random.randint(0, drop_margin)
        drop_end = random.randint(0, drop_margin)
        remaining_list = np.r_[drop_start:len(batch_img)-drop_end]
        batch_img = batch_img[remaining_list]
    return batch_img, remaining_list

def get_of_fisheye(H, W, center, magnitude):
    xx, yy = torch.linspace(-1, 1, W), torch.linspace(-1, 1, H)
    gridy, gridx = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack([gridx, gridy], dim=-1)
    d = center - grid
    d_sum = torch.sqrt((d**2).sum(axis=-1))
    grid += d * d_sum.unsqueeze(-1) * magnitude
    return grid  # Shape: (H, W, 2)

def RandomDistort(batch_img, max_magnitude):
    if random.random() > 0.5:
        num_frames, h, w, c = batch_img.shape
        center_x = (random.random() - 0.5) * 2
        center_y = random.random() * 0.25 - 1.5
        magnitude = random.random() * max_magnitude
        fisheye_grid = get_of_fisheye(h, w, torch.tensor([center_x, center_y]), magnitude)

        # Reshape batch_img to (batch_size, channels, height, width)
        batch_tensor = torch.FloatTensor(batch_img).permute(0, 3, 1, 2)

        # Expand fisheye_grid to match the batch size
        fisheye_grid = fisheye_grid.unsqueeze(0).repeat(num_frames, 1, 1, 1)

        # Apply grid_sample to all frames at once
        fisheye_output = F.grid_sample(batch_tensor, fisheye_grid, align_corners=False)

        # Reshape back to original format (frames, height, width, channels)
        return fisheye_output.permute(0, 2, 3, 1).numpy()
    else:
        return batch_img

def augment_lip_video(video):
    augmented_videos = []

    for _ in range(5):
        video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)

        flipped_video = TensorRandomFlip(video_tensor)

        # Adjust crop size based on input video size
        h, w = video_tensor.size(-2), video_tensor.size(-1)
        crop_size = (min(random.randint(88, 100), h), min(random.randint(88, 100), w))
        cropped_video = TensorRandomCrop(flipped_video, crop_size)

        aug_video = cropped_video.squeeze(0).permute(1, 2, 3, 0).numpy()

        aug_video, _ = RandomFrameDrop(aug_video, np.ones(len(aug_video)))

        aug_video = RandomDistort(aug_video, max_magnitude=0.1)

        # Adjust final crop size if necessary
        final_size = min(88, aug_video.shape[1], aug_video.shape[2])
        aug_video = CenterCrop(aug_video, (final_size, final_size))

        augmented_videos.append(aug_video)

    return augmented_videos

# Function to display sample frames
def display_sample_frames(original, augmented_list, num_frames=5):
    fig, axes = plt.subplots(6, num_frames, figsize=(20, 24))

    for i in range(num_frames):
        frame_idx = i * len(original) // num_frames

        axes[0, i].imshow(original[frame_idx])
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original Frame {frame_idx}')

        for j, augmented in enumerate(augmented_list):
            axes[j+1, i].imshow(augmented[frame_idx])
            axes[j+1, i].axis('off')
            axes[j+1, i].set_title(f'Augmented {j+1} Frame {frame_idx}')

    plt.tight_layout()
    plt.show()

# Test augmentation on a single video
def test_single_video(video_path):
    print(f"Testing augmentation on: {video_path}")

    # Load video
    original_video = load_video(video_path)

    # Generate augmentations
    augmented_videos = augment_lip_video(original_video)

    # Display sample frames from original and all augmented videos
    print("Displaying sample frames:")
    display_sample_frames(original_video, augmented_videos)

# Function to generate augmentations for all letters
def augment_all_letters(base_path):
    for letter_folder in os.listdir(base_path):
        letter_path = os.path.join(base_path, letter_folder)
        if os.path.isdir(letter_path):
            print(f"Processing letter: {letter_folder}")

            # Get the first video in the folder
            video_files = [f for f in os.listdir(letter_path) if f.endswith('.mp4')]
            if video_files:
                video_file = video_files[0]
                video_path = os.path.join(letter_path, video_file)
                print(f"  Processing video: {video_file}")

                # Load video
                original_video = load_video(video_path)

                # Generate augmentations
                augmented_videos = augment_lip_video(original_video)

                # Display sample frames from original and all augmented videos
                print("  Displaying sample frames:")
                display_sample_frames(original_video, augmented_videos)

                # Here you can add code to save the augmented videos if needed

            else:
                print(f"  No MP4 files found in {letter_folder}")

            print()  # Add a blank line between letters for readability

    print("Processing complete!")

# Test on the specific video
test_video_path = '/content/drive/MyDrive/AnasLips/ن/ن1.mp4'
test_single_video(test_video_path)

# Uncomment the following line when you're ready to process all letters
# augment_all_letters(base_path)
