import random
import numpy as np
import torch
import torch.nn.functional as F

def augment_lip_video(video):
    """
    Takes a cropped lip video as input and returns 5 augmented versions of that video.
    
    Args:
    video (numpy.ndarray): Input video of shape (frames, height, width, channels)
    
    Returns:
    list: List of 5 augmented versions of the input video
    """
    augmented_videos = []
    
    for _ in range(5):
        # Convert video to tensor for some operations
        video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
        
        # 1. Random Horizontal Flip
        flipped_video = TensorRandomFlip(video_tensor)
        
        # 2. Random Crop
        crop_size = (random.randint(88, 100), random.randint(88, 100))
        cropped_video = TensorRandomCrop(flipped_video, crop_size)
        
        # Convert back to numpy for other operations
        aug_video = cropped_video.squeeze(0).permute(1, 2, 3, 0).numpy()
        
        # 3. Random Frame Drop
        aug_video, _ = RandomFrameDrop(aug_video, np.ones(len(aug_video)))
        
        # 4. Random Distort
        aug_video = RandomDistort(aug_video, max_magnitude=0.1)
        
        # 5. Center Crop to ensure consistent size
        aug_video = CenterCrop(aug_video, (88, 88))
        
        augmented_videos.append(aug_video)
    
    return augmented_videos

# Helper functions (as provided in the original code)
def TensorRandomFlip(tensor):
    if random.random() > 0.5:
        return torch.flip(tensor, dims=[4])
    return tensor

def TensorRandomCrop(tensor, size):
    h, w = tensor.size(-2), tensor.size(-1)
    th, tw = size
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return tensor[:,:,:,y1:y1+th, x1:x1+tw]

def CenterCrop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    x1 = int(round((w - tw))/2.)
    y1 = int(round((h - th))/2.)    
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
    return grid.unsqueeze(0)

def RandomDistort(batch_img, max_magnitude): 
    if random.random() > 0.5:
        w, h = batch_img.shape[2], batch_img.shape[1]
        center_x = (random.random() - 0.5) * 2
        center_y = random.random() * 0.25 - 1.5
        magnitude = random.random() * max_magnitude
        fisheye_grid = get_of_fisheye(h, w, torch.tensor([center_x, center_y]), magnitude)
        fisheye_output = F.grid_sample(torch.FloatTensor(batch_img[None,...]), fisheye_grid, align_corners=False)
        return np.array(fisheye_output[0])
    else:
        return batch_img
