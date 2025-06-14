from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import pickle
import io

from googleapiclient.discovery import build

def get_actor_image(actor_name: str) -> str:
    service = build('customsearch', 'v1', developerKey="GOOGLE_DEVELOPER_API")

    query = actor_name.replace('_', ' ') + " bollywood actor portrait"
    
    try:
        res = service.cse().list(
            q=query,
            cx="SEARCH ENGINE ID",
            searchType="image",
            num=5,
            imgType="face",  # prioritize faces
            safe="active"
        ).execute()

        # Check top 5 results
        if "items" in res:
            for item in res["items"]:
                link = item.get("link", "")
                if link.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return link
            return res["items"][0]["link"]  # fallback to first
        else :
            print("No image found for", query)
    except Exception as e:
        print(f"[ERROR] Google image search failed: {e}")
    
    return ""

# -----------------------------
# Load Vocabulary
# -----------------------------
with open("vocab.json", "r") as f:
    vocab = json.load(f)
idx_to_word = {v: k for k, v in vocab.items()}  # Reverse mapping

# -----------------------------
# Model Definitions
# -----------------------------
class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super(CNNEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.cnn(images)  # Shape: [B, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # Now: [B, 2048]
        features = self.fc(features)  # Now: [B, embed_size]
        return self.relu(self.dropout(features))

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(LSTMDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        if features.dim() == 1:
            features = features.unsqueeze(0)  # Ensure shape [B, embed_size]
        embeddings = self.embed(captions[:, :-1])  # Remove <EOS>
        features = features.unsqueeze(1)  # [B, 1, embed_size]
        inputs = torch.cat((features, embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.fc(hiddens)
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# -----------------------------
# Load Model
# -----------------------------
app = Flask(__name__)
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)
num_layers = 3

encoder = CNNEncoder(embed_size)
decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers)
model = ImageCaptioningModel(encoder, decoder)

with torch.serialization.safe_globals([ImageCaptioningModel, CNNEncoder, LSTMDecoder]):
    model = torch.load("ImageCaptioningModel.pth", map_location="cpu", weights_only=False)

model.eval()

# -----------------------------
# Load Celebrity Features
# -----------------------------
with open("celebrity_features_cpu.pkl", "rb") as f:
    celebrity_features = pickle.load(f)

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Caption Generation Function
# -----------------------------
def generate_caption(image, model, vocab, idx_to_word, max_length=20):
    model.eval()
    with torch.no_grad():
        features = model.encoder(image.unsqueeze(0))  # [1, embed_size]
        if features.dim() == 1:
            features = features.unsqueeze(0)

        caption = [vocab["<SOS>"]]
        for _ in range(max_length):
            inputs = torch.tensor(caption).unsqueeze(0)  # [1, seq_len]
            outputs = model.decoder(features, inputs)  # [1, seq_len, vocab_size]
            predicted = outputs.argmax(2)[:, -1].item()
            if predicted == vocab["<EOS>"]:
                break
            caption.append(predicted)

        words = [idx_to_word.get(idx) for idx in caption[1:] if idx != vocab["<EOS>"]]
        return " ".join(words)

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(image_tensor, model):
    model.eval()
    with torch.no_grad():
        features = model.encoder(image_tensor.unsqueeze(0)).squeeze(0)
        return features / features.norm(p=2)

# -----------------------------
# Celebrity Matching
# -----------------------------
def match_with_celebrity(test_features, celebrity_features):
    best_match, highest_similarity = None, -1
    for name, features in celebrity_features.items():
        similarity = torch.dot(test_features, features).item()
        if similarity > highest_similarity:
            best_match, highest_similarity = name, similarity
    return best_match, highest_similarity

# -----------------------------
# Test Locally
# -----------------------------
# image = Image.open("face.jpg").convert("RGB")
# image_tensor = transform(image)

# caption = generate_caption(image_tensor, model, vocab, idx_to_word)
# test_features = extract_features(image_tensor, model)
# match_name, similarity = match_with_celebrity(test_features, celebrity_features)

# print("Generated Caption:", caption)
# print("Matched Celebrity:", match_name)
# print("Similarity Score:", round(similarity, 4))

# -----------------------------
# Flask Route (Uncomment to Use as API)
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")
    image_tensor = transform(image)

    caption = generate_caption(image_tensor, model, vocab, idx_to_word)
    test_features = extract_features(image_tensor, model)
    match_name, similarity = match_with_celebrity(test_features, celebrity_features)
    matchImgUrl = get_actor_image(match_name)

    return jsonify({
        "caption": caption,
        "matched_celebrity": match_name,
        "similarity_score": round(similarity, 4),
        "actor_image_url":matchImgUrl
    })

if __name__ == '__main__':
    app.run(debug=True)
