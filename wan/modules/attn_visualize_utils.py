import os
import math
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_attn_logit_metric(attn_logit):
    # import pdb; pdb.set_trace()
    batch_size, num_heads, L, L = attn_logit.shape
    max_attn_logits = torch.stack([torch.max(attn_logit[0][j]) for j in range(num_heads)])
    min_attn_logits = torch.stack([torch.min(attn_logit[0][j]) for j in range(num_heads)])
    mean_attn_logits = attn_logit.mean()
    var_attn_logits = attn_logit.var()
    return {"max":max_attn_logits, "min": min_attn_logits, "mean": mean_attn_logits, "var": var_attn_logits}

def get_relative_difference(attn_logits_ori, attn_logits_shift, epsilon=1e-9, head_chunk=1):
    batch_size, num_heads, L, L = attn_logits_ori.shape
    norm_diff_sum = 0
    norm_ori_sum = 0
    for i in range(0, num_heads, head_chunk):
        difference = attn_logits_shift[:, i:i+head_chunk] - attn_logits_ori[:, i:i+head_chunk] 
        norm_diff = torch.norm(difference) ** 2
        norm_ori = torch.norm(attn_logits_ori[:, i:i+head_chunk]) ** 2

        norm_diff_sum += norm_diff
        norm_ori_sum += norm_ori

    relative_change =torch.sqrt(norm_diff_sum) / torch.sqrt(norm_ori_sum)
    print(relative_change)
    return relative_change

def get_absolute_difference(attn_logits_ori, attn_logits_shift, head_chunk=1):
    batch_size, num_heads, L, L = attn_logits_ori.shape
    norm_diff_sum = 0
    for i in range(0, num_heads, head_chunk):
        difference = attn_logits_shift[:, i:i+head_chunk] - attn_logits_ori[:, i:i+head_chunk] 
        norm_diff = torch.norm(difference) ** 2

        norm_diff_sum += norm_diff
    relative_change =torch.sqrt(norm_diff_sum)
    print(relative_change)
    return relative_change



class AttentionMapController:
    def __init__(self, save_dir, text_length, height, width, thres=0.1):
        self.save_dir = save_dir
        self.text_length = text_length
        self.height = height
        self.width = width
        # self.compressed_num_frames = compressed_num_frames
        self.thres = thres
        self.cross_attn_sum = None
        self.cross_attn_count = 0
        self.video_self_attn_sum = None
        self.video_self_attn_count = 0
        self.self_attn_mask = None
        print(f"Set mask save directory for AttentionMapController: {save_dir}")

    def get_self_attn_mask(self):
        return self.self_attn_mask
    def get_cross_attn_sum(self):
        return self.cross_attn_sum
    def get_cross_attn_count(self):
        return self.cross_attn_count
    
    def set_self_attn_mask(self, mask):
        self.self_attn_mask = mask

    def reset_cross_attns(self):
        self.cross_attn_sum = None
        self.cross_attn_count = 0
    def reset_video_self_attn(self):
        self.video_self_attn_sum = None
        self.video_self_attn_count = 0
    
    def reset_self_attn_mask(self):
        self.self_attn_mask = None
    
    def save_cur_cross_attn_map(self, q, k):
        batch_size, num_heads, seq_len, head_dim = q.shape
        _, _, text_len, _ = k.shape
        # Initialize result tensor (on GPU)
        attn_map_mean = torch.zeros(batch_size, seq_len, text_len, device=q.device, dtype=q.dtype)
        
        # Parameters for batch computation
        batch_chunk = 1  # Number of batches to process at a time
        head_chunk = 1   # Number of attention heads to process at a time, can be adjusted based on GPU memory
        for i in range(0, batch_size, batch_chunk):
            for j in range(0, num_heads, head_chunk):
                # Select data for current batch and attention heads
                q_chunk = q[i:i+batch_chunk, j:j+head_chunk]
                k_chunk = k[i:i+batch_chunk, j:j+head_chunk]
                
                # Calculate attention scores for current attention heads
                attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)
                attn_probs = torch.softmax(attn_scores, dim=-1)
                
                # Add to mean attention map
                attn_map_mean[i:i+batch_chunk] += attn_probs.sum(dim=1)
        
        # Calculate mean
        attn_map_mean /= num_heads
        video_to_text_attn = attn_map_mean

        # Update accumulated sum and count
        if self.cross_attn_sum is None:
            self.cross_attn_sum = video_to_text_attn
        else:
            self.cross_attn_sum += video_to_text_attn
        self.cross_attn_count += 1
    
    def save_cur_self_attn_map(self, q, k):
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Initialize result tensor (on GPU)
        attn_map_mean = torch.zeros(batch_size, seq_len, seq_len, device=q.device, dtype=q.dtype)
        
        # Parameters for batch computation
        batch_chunk = 1  # Number of batches to process at a time
        head_chunk = 1   # Number of attention heads to process at a time, can be adjusted based on GPU memory
        for i in range(0, batch_size, batch_chunk):
            for j in range(0, num_heads, head_chunk):
                # Select data for current batch and attention heads
                q_chunk = q[i:i+batch_chunk, j:j+head_chunk]
                k_chunk = k[i:i+batch_chunk, j:j+head_chunk]
                
                # Calculate attention scores for current attention heads
                attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)
                attn_probs = torch.softmax(attn_scores, dim=-1)
                
                # Add to mean attention map
                attn_map_mean[i:i+batch_chunk] += attn_probs.sum(dim=1)
        
        # Calculate mean
        attn_map_mean /= num_heads
        # video_to_text_attn = attn_map_mean[:, self.text_length:, :self.text_length]

        # # Update accumulated sum and count
        # if self.cross_attn_sum is None:
        #     self.cross_attn_sum = video_to_text_attn
        # else:
        #     self.cross_attn_sum += video_to_text_attn
        # self.cross_attn_count += 1
        
        # Process video self attention
        video_to_video_attn = attn_map_mean
        if self.video_self_attn_sum is None:
            self.video_self_attn_sum = video_to_video_attn
        else:
            self.video_self_attn_sum += video_to_video_attn
        self.video_self_attn_count += 1

    def aggregate_cross_attn_map(self, token_idx=[1]):
        if self.cross_attn_sum is None or self.cross_attn_count == 0:
            return None

        attn_map = self.cross_attn_sum / self.cross_attn_count
        # import pdb;pdb.set_trace()
        B, HWF, T = attn_map.shape
        F = HWF // (self.height * self.width)
        attn_map = attn_map.reshape(B, F, self.height, self.width, T)

        # if isinstance(token_idx, (list, ListConfig)):
        if isinstance(token_idx, list):
            attn_map = attn_map[..., token_idx]
            attn_map = attn_map.sum(dim=-1)  # Sum over selected tokens
        else:
            attn_map = attn_map[..., token_idx:token_idx+1].squeeze(-1)

        # Get min and max values using PyTorch
        attn_min = attn_map.amin(dim=(2, 3), keepdim=True)
        attn_max = attn_map.amax(dim=(2, 3), keepdim=True)
        
        normalized_attn = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)  # Add small value to avoid division by zero
        
        return normalized_attn
    
    def visualize_video_video_spatial_attn_map(self, step_idx, layer_idx, frame_idx=None,
                                   show_title=True, show_labels=True, colormap='viridis'):
        """
        Visualize spatial attention map (H*W, H*W)
        
        Args:
            step_idx: int, current step
            layer_idx: int, current layer
            frame_idx: int or None, specified frame index
            show_title: bool, whether to show title
            show_labels: bool, whether to show axis labels
            colormap: str, colormap to use
        """
        
        if self.video_self_attn_sum is None or self.video_self_attn_count == 0:
            return

        # Get mean attention map
        attn_map = self.video_self_attn_sum / self.video_self_attn_count
        B, HWF, HWF = attn_map.shape
        F = HWF // (self.height * self.width)
        HW = self.height * self.width
        
        # Reshape to [B, F, HW, F, HW]
        attn_map = attn_map.reshape(B, F, HW, F, HW)

        # Create save directory
        save_path = os.path.join(self.save_dir, f"step{step_idx}_layer{layer_idx}")
        os.makedirs(save_path, exist_ok=True)

        # Determine frames to process
        if frame_idx is not None:
            frames_to_process = [frame_idx]
        else:
            frames_to_process = range(F)

        # Create attention map visualization for each frame
        for f_idx in frames_to_process:
            # Get spatial attention for current frame [HW, HW]
            spatial_attn = attn_map[0, f_idx, :, f_idx, :]
            
            # Normalize
            attn_min = spatial_attn.min()
            attn_max = spatial_attn.max()
            print(attn_max)
            print(attn_min)
            normalized_attn = (spatial_attn - attn_min) / (attn_max - attn_min + 1e-6)

            # Create figure
            plt.figure(figsize=(10, 10))
            plt.imshow(normalized_attn.cpu().numpy(), cmap=colormap)
            plt.colorbar()
            
            if show_labels:
                plt.xlabel('Spatial Position')
                plt.ylabel('Spatial Position')
            else:
                plt.xticks([])
                plt.yticks([])
            
            if show_title:
                plt.title(f'Spatial Attention Map (Frame {f_idx})')
            # Save image
            save_name = f'spatial_attn_map_frame{f_idx}.png'
            plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
            plt.close()

            # Optional: save as numpy array for later analysis
            np.save(os.path.join(save_path, f'spatial_attn_map_frame{f_idx}.npy'), 
                    normalized_attn.cpu().numpy())
            

    def visualize_video_video_temporal_attn_map(self, step_idx, layer_idx, token_h, token_w,
                                    show_title=True, show_labels=True, colormap='viridis'):
        """
        Visualize frame-to-frame attention map (F, F) at specified spatial position
        
        Args:
            step_idx: int, current step
            layer_idx: int, current layer
            token_h: int, token height position
            token_w: int, token width position
        """
        if self.video_self_attn_sum is None or self.video_self_attn_count == 0:
            return

        # Get mean attention map
        attn_map = self.video_self_attn_sum / self.video_self_attn_count
        B, HWF, HWF = attn_map.shape
        F = HWF // (self.height * self.width)
        HW = self.height * self.width
        
        # Reshape to [B, F, HW, F, HW]
        attn_map = attn_map.reshape(B, F, HW, F, HW)
        
        # Calculate token index in HW
        token_idx = token_h * self.width + token_w
        
        # Extract frame-to-frame attention at this position [F, F]
        temporal_attn = attn_map[0, :, token_idx, :, token_idx]
        
        # Normalize
        attn_min = temporal_attn.min()
        attn_max = temporal_attn.max()
        normalized_attn = (temporal_attn - attn_min) / (attn_max - attn_min + 1e-6)

        # Create save directory
        save_path = os.path.join(self.save_dir, f"step{step_idx}_layer{layer_idx}")
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(8, 8))
        plt.imshow(normalized_attn.cpu().numpy(), cmap=colormap)
        plt.colorbar()
        
        if show_labels:
            plt.xlabel('Frame Index')
            plt.ylabel('Frame Index')
        else:
            plt.xticks([])
            plt.yticks([])
        
        if show_title:
            plt.title(f'Temporal Attention Map at Position ({token_h}, {token_w})')
            
        # Save image
        save_name = f'temporal_attn_map_pos_{token_h}_{token_w}.png'
        plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
        plt.close()

        # Save numpy array
        np.save(os.path.join(save_path, f'temporal_attn_map_pos_{token_h}_{token_w}.npy'), 
                normalized_attn.cpu().numpy())
        
        return normalized_attn   
    
    def visualize_video_video_attn_map(self, step_idx, layer_idx,
                                    show_title=True, show_labels=True, colormap='viridis'):
        """
        Visualize frame-to-frame attention map (F, F) at specified spatial position
        
        Args:
            step_idx: int, current step
            layer_idx: int, current layer
            token_h: int, token height position
            token_w: int, token width position
        """
        if self.video_self_attn_sum is None or self.video_self_attn_count == 0:
            return

        # Get mean attention map
        attn_map = self.video_self_attn_sum / self.video_self_attn_count
        B, HWF, HWF = attn_map.shape

        # Create save directory
        save_path = os.path.join(self.save_dir, f"step{step_idx}_layer{layer_idx}")
        os.makedirs(save_path, exist_ok=True)
        
        attn_map = attn_map[0]
        # Normalize
        attn_min = attn_map.min()
        attn_max = attn_map.max()
        normalized_attn = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)

        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(normalized_attn.cpu().numpy(), cmap=colormap)
        plt.colorbar()
        
        if show_labels:
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
        else:
            plt.xticks([])
            plt.yticks([])
        
        if show_title:
            plt.title(f'Spatial Temporal Attention Map')
        # Save image
        save_name = f'spatial_temporal_attn_map.png'
        plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
        plt.close()

        # # Optional: save as numpy array for later analysis
        # np.save(os.path.join(save_path, f'spatial_attn_map_frame{f_idx}.npy'), 
        #         normalized_attn.cpu().numpy())

    def visualize_token_attention_masks(self, token_indices, step, layer, frame_indices, num_tokens_to_show=6, layout='vertical'):
        """
        Generate and save token attention visualization results directly
        
        Args:
            token_indices: List[List[int]] - List of token indices, e.g. [[0], [1], [2], [0,1,2]]
            step: int - Current step
            layer: int - Current layer
            frame_indices: list - List of frame indices to visualize
            num_tokens_to_show: int - Number of tokens to display
            layout: str - Layout mode, 'horizontal' or 'vertical'
        """
        def apply_magma_colormap(image):
            """
            Convert grayscale image to magma colormap
            """
            img_array = np.array(image)
            img_normalized = img_array / 255.0
            colored_array = plt.cm.magma(img_normalized)
            colored_image = Image.fromarray((colored_array[:, :, :3] * 255).astype(np.uint8))
            return colored_image
        
            
        # Only process specified number of tokens
        token_indices = token_indices[:num_tokens_to_show]
        num_tokens = len(token_indices)
        
        for frame_idx in frame_indices:
            print(f"Processing step {step}, layer {layer}, frame {frame_idx}")
            
            # Create image layout
            if layout == 'horizontal':
                fig, axes = plt.subplots(1, num_tokens, figsize=(2*num_tokens, 2))
                subplot_adjust = {'wspace': 0.05, 'hspace': 0}
            else:  # vertical
                fig, axes = plt.subplots(num_tokens, 1, figsize=(2, 1.2*num_tokens))
                subplot_adjust = {'wspace': 0, 'hspace': 0.01}
            
            if num_tokens == 1:
                axes = [axes]
                
            # Set background color
            fig.patch.set_facecolor('white')
            
            # Generate and display attention map for each token/token combination
            for idx, token_idx in enumerate(token_indices):
                # Generate mask
                mask = self.aggregate_cross_attn_map(token_idx=token_idx)
                mask_target = mask[-1]  # (F, H, W)
                
                # Extract specified frame
                frame_mask = mask_target[frame_idx]  # (H, W)
                
                # Convert to PIL Image and apply magma colormap
                frame_img = Image.fromarray((frame_mask.cpu().numpy() * 255).astype(np.uint8))
                colored_frame = apply_magma_colormap(frame_img)
                
                # Display image
                axes[idx].imshow(np.array(colored_frame))
                axes[idx].axis('off')
                
            # Adjust subplot spacing
            plt.subplots_adjust(**subplot_adjust)
            
            # Tight layout
            if layout == 'vertical':
                plt.tight_layout(pad=0.1, h_pad=0.1)
                
            # Save image
            save_path = os.path.join(self.save_dir, f"step{step}_layer{layer}")
            os.makedirs(save_path, exist_ok=True)
            
            layout_str = 'horizontal' if layout == 'horizontal' else 'vertical'
            save_path = os.path.join(
                save_path,
                f'token_attention_{layout_str}_'
                f'step_{step}_layer_{layer}_'
                f'selected_frame_{frame_idx}_'
                f'first_{num_tokens_to_show}_tokens.png'
            )
            
            plt.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0, facecolor='white')
            plt.close()
            
            print(f"Token attention visualization saved to {save_path}")
