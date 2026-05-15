import gradio as gr
import torch
import torch.nn as nn
import json
from PIL import Image
from torchvision import transforms

# ── Model Architecture ذ ──────────────────────────────────
class VGG_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn_block_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.neural_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14 * 14 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.cnn_block_1(x)
        x = self.cnn_block_2(x)
        x = self.cnn_block_3(x)
        x = self.cnn_block_4(x)
        x = self.neural_network(x)
        return x


# ── Load class names & model ──────────────────────────────────────────────────
with open(r'E:\Computer science\python\Computer Vision\Youssef work\Deploy\class_names.json') as f:
    class_names = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = VGG_CNN(num_classes=len(class_names))
model.load_state_dict(torch.load( r'E:\Computer science\python\Computer Vision\Youssef work\Deploy\model_from_scratch.pth', map_location=device))
model.to(device)
model.eval()
print(f"Model loaded on {device} | Classes: {len(class_names)}")

# ── Image transform ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Treatment database ────────────────────────────────────────────────────────

TREATMENT = {
    "Pepper__bell___Bacterial_spot":    {"treatment": "Apply copper bactericide spray. Remove and destroy severely infected plants."},
    "Pepper__bell___healthy":           {"treatment": "Plant is healthy. Use certified seed and proper spacing."},
    "Potato___Early_blight":            {"treatment": "Remove infected lower leaves. Apply fungicide when disease first appears."},
    "Potato___Late_blight":             {"treatment": "Destroy infected plants immediately. Apply appropriate fungicide preventively."},
    "Potato___healthy":                 {"treatment": "Plant is healthy. Use certified seed potatoes and practice crop rotation."},
    "Tomato_Bacterial_spot":            {"treatment": "Apply copper-based bactericide. Remove infected tissue."},
    "Tomato_Early_blight":              {"treatment": "Remove lower infected leaves. Apply fungicide."},
    "Tomato_Late_blight":               {"treatment": "Remove and destroy infected plants. Apply fungicide preventively."},
    "Tomato_Leaf_Mold":                 {"treatment": "Improve greenhouse ventilation. Apply fungicide."},
    "Tomato_Septoria_leaf_spot":        {"treatment": "Remove infected leaves immediately. Apply fungicide."},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"treatment": "Apply insecticidal soap or neem oil."},
    "Tomato__Target_Spot":              {"treatment": "Apply appropriate fungicide. Remove infected plant material."},
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {"treatment": "No cure — remove infected plants. Control whitefly vectors."},
    "Tomato__Tomato_mosaic_virus":      {"treatment": "Remove infected plants. Disinfect tools with bleach solution."},
    "Tomato_healthy":                   {"treatment": "Plant is healthy. Regular scouting and balanced fertilisation."},
}

# ── Prediction function ───────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "Please upload an image.", "", ""

    img    = Image.fromarray(image).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top3_probs, top3_idx = torch.topk(probs, 3)

    best_cls  = class_names[top3_idx[0].item()]
    best_conf = top3_probs[0].item() * 100
    info      = TREATMENT.get(best_cls, {"treatment": "Consult a local agronomist."})
    treatment = info["treatment"]

    clean_name = best_cls.replace("___", " - ").replace(",_", " ").replace("_", " ")
    if "healthy" in best_cls:
        status = f"## Plant is Healthy\n\n**{clean_name}**\n\nConfidence: **{best_conf:.1f}%**"
    else:
        status = f"## Disease Detected\n\n**{clean_name}**\n\nConfidence: **{best_conf:.1f}%**"

    medals   = ["1st", "2nd", "3rd"]
    top3_str = ""
    for i in range(3):
        cls  = class_names[top3_idx[i].item()]
        conf = top3_probs[i].item() * 100
        name = cls.replace("___", " - ").replace("_", " ")
        top3_str += f"{medals[i]}  {name}  -  {conf:.1f}%\n"

    return status, treatment, top3_str


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="green"),
    title="Plant Disease Detector"
) as demo:

    gr.Markdown("""
    # Plant Disease Detection System
    ### Upload a leaf image — the system will identify the disease and recommend treatment
    **Model:** CNN VGG-style | **Dataset:** PlantVillage | **Val Accuracy: 97.51%**
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(label="Upload Leaf Image", type="numpy", height=320)
            btn       = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            out_status    = gr.Markdown()
            out_treatment = gr.Textbox(label="Treatment Recommendation", lines=4, interactive=False)
            out_top3      = gr.Textbox(label="Top 3 Predictions", lines=4, interactive=False)

    btn.click(
        fn      = predict,
        inputs  = img_input,
        outputs = [out_status, out_treatment, out_top3]
    )

    gr.Markdown("""
    ---
    > Trained on PlantVillage Dataset | CNN VGG-style (PyTorch)
    """)

demo.launch(share=True)