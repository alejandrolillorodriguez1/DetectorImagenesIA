from fastapi import FastAPI, UploadFile , File
from io import BytesIO
from PIL import Image
from torchvision import transforms
from training.src.model import CNNDetector
import torch


app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/best_model.pth"

allowed_extensions = ["jpg", "jpeg", "png"]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

def load_model():
    model = CNNDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()
    return model

model = load_model()

@app.post("/predict")
async def predict(image : UploadFile = File(...)):
    if image.filename.split(".")[-1].lower() not in allowed_extensions:
        return {"error": "Invalid file type. Only jpg, jpeg, and png are allowed."}

    image_data = await image.read()
    pil_image = Image.open(BytesIO(image_data)).convert("RGB")

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_index = outputs.argmax(
            dim=1
        ).item()

    confidence = probabilities[
            0,
            predicted_index
        ].item()

    class_names = ["FAKE", "REAL"]

    predicted_class = class_names[
            predicted_index
    ]
    return {
        
        "filename": image.filename,
        "size_bytes": len(image_data),
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2)
    }