from PIL import Image
import torch
# import torchvision.models as models
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

#Faster-RCNN object detection


detector = fasterrcnn_resnet50_fpn(pre_trained=True).eval().cuda()

transform_detect = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def detect_objects(image):
    image = transform_detect(image).unsqueeze(0).cuda()
    with torch.no_grad():
        predictions = detector(image)

    return predictions



def extract_objects_from_frames(dataloader):
    scene_graphs = []

    for i, (frame, _) in enumerate(dataloader):
        frame = frame.squeeze().permute(1, 2, 0).cpu().numpy()  
        frame_pil = Image.fromarray((frame * 255).astype('uint8'))  

        detections = detect_objects(frame_pil)  
        boxes = detections[0]['boxes'].cpu().numpy()  
        labels = detections[0]['labels'].cpu().numpy()  

        scene_graph = {'frame_id': i, 'objects': [], 'edges': []}

        for j, (box, label) in enumerate(zip(boxes, labels)):
            scene_graph['objects'].append({'id': j, 'bbox': box, 'label': label})

        scene_graphs.append(scene_graph)

    return scene_graphs