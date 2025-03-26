import os
import numpy as np
import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models.optical_flow as optical_flow
from tqdm import tqdm

data_dir = "/home/ayush/Desktop/Datasets/archive"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform = None, classes_to_keep = None):
        self.classes_to_keep = classes_to_keep
        super().__init__(root, transform=transform)
    
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        if self.classes_to_keep is not None:
            classes = [c for c in classes if c in self.classes_to_keep]
            class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx

print("Loading Dataset...")
train_set = FilteredImageFolder(root = os.path.join(data_dir, "Train/"), transform=transform, classes_to_keep=["Assault", "Shooting"])
# test_set = FilteredImageFolder(root = os.path.join(data_dir, "Test/") , transform=transform) 

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#Feature extraction using ResNet-18
print("\nExtracting features using Resnet-18...")

# Load pre-trained ResNet-18 and define the feature extractor
model = models.resnet18(pretrained=True)
feature_extractor = create_feature_extractor(model, return_nodes={'avgpool': 'features'})

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# Extract features
features = []
with torch.no_grad():
    for inputs, _ in tqdm(train_loader, desc="Extracting features"):
        inputs = inputs.to(device)
        outputs = feature_extractor(inputs)['features']  
        outputs = outputs.view(outputs.size(0), -1)      
        features.append(outputs)

# Concatenate all features into a single tensor
features = torch.cat(features, dim=0)
torch.save(features, 'extracted_features.pt')
print("Feature extraction complete! Saved to 'extracted_features.pt'")

#Optical Flow
output_folder = "optical_flow_results"
os.makedirs(output_folder, exist_ok=True)

# Load the RAFT model and move it to GPU
print("\nLoading RAFT optical flow model...")
raft_model = optical_flow.raft_large(pretrained=True)
raft_model = raft_model.eval().cuda()

def process_frame_pairs(dataloader):
    prev_frame = None

    for i, (frame, _) in enumerate(tqdm(dataloader, desc="Computing optical flow")):
        # Move frame to GPU
        
        frame = frame.cuda()

        if prev_frame is not None:
            with torch.no_grad():
                # Compute optical flow
                flow = raft_model(prev_frame, frame)[-1]  # Use the last refinement level

            # Move flow to CPU and convert to NumPy
            flow = flow.squeeze().cpu().numpy()

            # Save the optical flow to a file in the output folder
            flow_file = os.path.join(output_folder, f'optical_flow_{i-1}_{i}.npy')
            np.save(flow_file, flow)

        # Update prev_frame for the next iteration
        prev_frame = frame

# Process the frame pairs from the dataloader
print("\nProcessing optical flow...")
process_frame_pairs(train_loader)
print(f"Optical flow computation complete! Results saved in '{output_folder}'")



