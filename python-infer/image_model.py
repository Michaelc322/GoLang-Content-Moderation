import io
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights).eval()
transform = weights.transforms()
MODEL_VERSION = "mobilenetv3@imagenet-proxy"

@torch.no_grad()
def classify_image(img_bytes: bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)
    score = probs[0][1].item()
    flagged = score > 0.7
    return score, flagged, MODEL_VERSION
