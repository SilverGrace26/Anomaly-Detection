import numpy as np
import imageio
from flow_vis import flow_to_color
import os
import warnings

def create_flow_slideshow(flow_dir, output_file='flow_slideshow.mp4', fps=10):
    """Create slideshow from batch flow files"""
    try:
        # Get sorted list of flow files
        flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.npy')])
        
        if not flow_files:
            print("No .npy files found in directory")
            return False

        # Collect all frames
        images = []
        for flow_file in flow_files:
            flow_batch = np.load(os.path.join(flow_dir, flow_file))  # shape (32, 2, 224, 224)
            
            # Process each flow in the batch
            for i in range(flow_batch.shape[0]):
                flow = flow_batch[i].transpose(1, 2, 0)  # to (224, 224, 2)
                flow_color = flow_to_color(flow, convert_to_bgr=False)
                images.append((flow_color * 255).astype(np.uint8))

        if not images:
            print("No valid flow arrays found")
            return False

        # Try writing as MP4
        try:
            imageio.mimsave(output_file, images, fps=fps)
            print(f"Successfully saved MP4 to {output_file}")
            return True
        except Exception as e:
            print(f"MP4 save failed: {e}")
            print("Trying alternative formats...")
            
            # Fallback 1: Try GIF
            try:
                gif_file = os.path.splitext(output_file)[0] + '.gif'
                imageio.mimsave(gif_file, images, fps=fps)
                print(f"Saved as GIF to {gif_file}")
                return True
            except Exception as e:
                print(f"GIF save failed: {e}")
                
            # Fallback 2: Save as PNG sequence
            png_dir = os.path.splitext(output_file)[0] + '_frames'
            os.makedirs(png_dir, exist_ok=True)
            for i, img in enumerate(images):
                imageio.imwrite(os.path.join(png_dir, f'frame_{i:04d}.png'), img)
            print(f"Saved as PNG sequence to {png_dir}")
            return True

    except Exception as e:
        print(f"Error: {e}")
        return False

# Example usage
flow_directory = "optical_flow_results"
create_flow_slideshow(flow_directory)