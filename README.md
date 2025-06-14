# Face Feature Matching & Caption Generation using CNN-LSTM

This project combines deep learning techniques to automatically describe facial features from images and match them to the most visually similar celebrity. It leverages a **CNN encoder** for feature extraction and an **LSTM decoder** for caption generation, creating an end-to-end system that can both describe faces and find celebrity look-alikes.

## 🎯 Project Overview

The system performs two main tasks:
1. **Caption Generation**: Automatically generates descriptive text for facial features (e.g., "sharp jawline, thick eyebrows")
2. **Celebrity Matching**: Finds the most visually similar celebrity from a pre-trained database using cosine similarity

## 🏗️ Architecture

### **CNN Encoder**
- Uses a pre-trained **ResNet-50** model for feature extraction
- Removes the final classification layer and adds a custom fully connected layer
- Extracts high-level visual features like eye shape, jawline, and hair structure
- Output: Dense feature vector representing the face

### **LSTM Decoder**
- Takes image features as input and generates captions word by word
- Uses embedding layers to convert words to vectors
- Learns to associate visual information with descriptive language
- Trained with teacher forcing during training phase

### **Combined Model**
- CNN + LSTM form a complete Image Captioning System
- Loss Function: CrossEntropyLoss for word prediction
- Trained end-to-end on face images with corresponding descriptive captions

## 📊 Dataset Structure

dataset/
├── images/
│ ├── celebrity1.jpg
│ ├── celebrity2.jpg
│ └── test_image.jpg
└── captions.csv

**Caption CSV Format:**
image_name,caption
celebrity1.jpg,"sharp jawline, thick eyebrows, brown eyes"
celebrity2.jpg,"round face, thin lips, blonde hair"


## 🔄 Data Preprocessing Pipeline

### **Vocabulary Generation**
- Captions are lowercased and tokenized
- Special tokens added: `<SOS>` (Start of Sequence), `<EOS>` (End of Sequence)
- Custom vocabulary built with unique index for each word
- Saved as `vocab.json` for inference

### **Dataset Class**
- Loads images from folder and corresponding captions from CSV
- Applies image transformations (resize to 224x224, normalize)
- Converts captions to numerical sequences using vocabulary
- Returns image tensor and caption sequence pairs

### **DataLoader**
- Efficient batching with shuffling
- Parallel data loading for faster training
- Handles variable-length captions with padding

## 🧠 Model Training Process

### **Training Components**
| Component | Role | Importance |
|-----------|------|------------|
| CNN Encoder | Extracts features from images | Captures key visual traits of faces |
| LSTM Decoder | Generates captions from image features | Translates visual features to human-readable text |
| Vocabulary | Maps words to indices | Enables LSTM to understand and predict words |
| Cosine Similarity | Compares feature vectors | Identifies visually similar faces efficiently |

### **Training Steps**
1. **Feature Extraction**: CNN encoder processes face images
2. **Caption Generation**: LSTM decoder learns to generate descriptions
3. **Loss Calculation**: CrossEntropyLoss between predicted and actual captions
4. **Backpropagation**: Model weights updated using Adam optimizer
5. **Celebrity Feature Storage**: Pre-computed features saved for inference

## 🚀 System Architecture & Flow

### **Backend (Python Flask API)**

The main inference system consists of:

Model Architecture
class CNNEncoder(nn.Module):
# ResNet-50 based feature extractor

class LSTMDecoder(nn.Module):
# LSTM-based caption generator

class ImageCaptioningModel(nn.Module):
# Combined CNN-LSTM model


### **API Endpoint**
POST /predict
Content-Type: multipart/form-data
Body: image file


**Response:**
{
"caption": "sharp jawline, thick eyebrows",
"matched_celebrity": "actor_name",
"similarity_score": 0.8542,
"actor_image_url": "https://image_url.jpg"
}

### **Google Custom Search Integration**

The system uses **Google Custom Search API** to fetch celebrity images:

def get_actor_image(actor_name: str) -> str:
service = build('customsearch', 'v1', developerKey="YOUR_API_KEY")
query = actor_name.replace('_', ' ') + " bollywood actor portrait"
# Returns high-quality celebrity image URL

**Features:**
- Searches for high-quality portrait images
- Filters for face-focused images
- Safe search enabled
- Fallback mechanisms for reliability

## 🔧 Installation & Setup

### **Prerequisites**
Python 3.8+
PyTorch
Flask
PIL (Pillow)
Google API Client

### **Installation Steps**

1. **Clone the repository**
git clone <repository-url>
cd face-feature-matching


2. **Install dependencies**
pip install torch torchvision flask pillow google-api-python-client

3. **Download required files**
- `ImageCaptioningModel.pth` (trained model)
- `vocab.json` (vocabulary mapping)
- `celebrity_features_cpu.pkl` (pre-computed celebrity features)

4. **Set up Google API**
- Get Google Custom Search API key
- Create Custom Search Engine ID
- Update credentials in the code

### **Running the Application**

#### **Backend (Flask API)**
python app.py

Server runs on `http://localhost:5000`

#### **Frontend (React)**
cd frontend/front
npm install
npm start

#### **Backend API Server (Node.js)**
cd backend
npm install
npm start


## 🌐 Web Application Flow

### **User Journey**
1. **Upload Image**: User uploads a face image through React frontend
2. **API Processing**: Node.js backend forwards request to Flask ML service
3. **Feature Extraction**: CNN encoder extracts facial features
4. **Caption Generation**: LSTM decoder generates descriptive text
5. **Celebrity Matching**: Cosine similarity finds best celebrity match
6. **Image Retrieval**: Google Search API fetches celebrity image
7. **Response Display**: Results shown on frontend with caption and celebrity match

### **System Components**
React Frontend ↔ Node.js Backend ↔ Flask ML Service ↔ Google Search API
↓
PyTorch Models
(CNN + LSTM)

## 🎯 Key Features

✅ **End-to-end trainable image captioning model**  
✅ **Real-time celebrity look-alike matching**  
✅ **Automatic facial feature description generation**  
✅ **Google Search API integration for celebrity images**  
✅ **Web-based interface for easy interaction**  
✅ **RESTful API architecture**  
✅ **Scalable and extensible design**

## 🔮 Future Enhancements

- **Transformer Integration**: Replace LSTM with Transformer architecture for better performance
- **Facial Landmark Detection**: Add precise facial feature localization
- **Real-time Processing**: Implement WebSocket for live video processing
- **Mobile Application**: Develop native mobile apps
- **Enhanced Dataset**: Expand celebrity database and feature descriptions
- **Multi-language Support**: Generate captions in multiple languages

## 🛠️ Technical Specifications

- **Framework**: PyTorch for deep learning, Flask for API
- **Model**: ResNet-50 + LSTM architecture
- **Image Processing**: 224x224 RGB normalization
- **Similarity Metric**: Cosine similarity for feature matching
- **API Integration**: Google Custom Search for image retrieval
- **Frontend**: React.js with responsive design
- **Backend**: Node.js with Express framework