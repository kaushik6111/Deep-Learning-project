import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from dataset import MEAN, STD

class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient):
        self.gradients = gradient

    def generate(self, image_tensor):
        self.model.eval()
        with torch.enable_grad():
            logits = self.model(image_tensor)
            self.model.zero_grad(set_to_none=True)
            logits[0, 0].backward()
        weights = self.gradients.detach().mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations.detach()).sum(dim=1)[0]
        heatmap = torch.relu(heatmap).cpu().numpy()
        maximum = float(heatmap.max())
        if maximum > 0:
            heatmap /= maximum
        return heatmap, torch.sigmoid(logits[0, 0]).item()

    def close(self):
        self.handle.remove()

def save_gradcam(model, image_path, device, output_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tensor = preprocess(image).unsqueeze(0).to(device)
    cam = GradCAM(model, model.backbone.features.norm5)
    try:
        heatmap, probability = cam.generate(tensor)
    finally:
        cam.close()
    original = np.array(image.resize((224, 224)))
    resized = cv2.resize(heatmap, (224, 224))
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.6, colored, 0.4, 0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, arr, title in zip(axes, [original, resized, overlay], ['Input X-Ray', 'Grad-CAM', f'Pneumonia score: {probability:.1%}']):
        ax.imshow(arr, cmap='jet' if title == 'Grad-CAM' else None)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Grad-CAM saved: {output_path}')
