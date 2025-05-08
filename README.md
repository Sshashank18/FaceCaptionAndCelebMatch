# 🧠 Face Feature Matching & Caption Generation using CNN-LSTM

This project combines deep learning techniques to automatically describe facial features from images and match them to the most visually similar celebrity. It leverages a **CNN encoder** for feature extraction and an **LSTM decoder** for caption generation.

---

## 📁 Dataset Structure

- A **folder of images** containing faces (celebrities + test images).
- A **CSV file** with corresponding captions describing features for each image.
  - Example caption: `"sharp jawline, thick eyebrows"`

---

## 🔄 Data Preprocessing

### 🧾 Vocabulary Generation
- Captions are:
  - Lowercased and tokenized.
  - Converted into sequences of token IDs.
- A custom vocabulary is built, assigning a unique index to each word.

### 🧷 Dataset Class
- Loads:
  - Image from folder.
  - Corresponding caption from the CSV.
- Transforms:
  - Applies image preprocessing (resize, normalize).
- Returns:
  - Image tensor and numerical caption sequence.

### 🧪 DataLoader
- Wraps the dataset for:
  - Efficient batching
  - Shuffling and parallel loading

---

## 🧠 Model Architecture

### 🖼️ CNN Encoder
- Uses a pre-trained CNN (e.g., ResNet) to extract **deep feature vectors** from input face images.
- These vectors capture high-level features like:
  - Eye shape
  - Jawline
  - Hair structure

### ✏️ LSTM Decoder
- Receives image features as input.
- Generates a **caption sequence** word by word.
- Learns to associate visual information with descriptive language.

### 🔗 Combined Model
- CNN + LSTM form a complete **Image Captioning System**.
- Loss Function: `CrossEntropyLoss` for training word prediction.

---

## 🚀 Inference Pipeline

### 1. **Generate Caption**
- A test image is passed through the trained CNN → LSTM model.
- The model outputs a caption describing facial features.

### 2. **Find Celebrity Match**
- Both test and celebrity images are encoded using the **CNN encoder**.
- **Cosine Similarity** is computed between their feature vectors.
- The celebrity with the **highest similarity score** is returned as the closest match.

---

## ⚙️ Component Importance

| Component         | Role                                  | Why It's Important                                   |
|------------------|---------------------------------------|------------------------------------------------------|
| `CNN Encoder`     | Extracts features from images         | Captures key visual traits of the face               |
| `LSTM Decoder`    | Generates caption from image features | Translates visual features to human-readable format  |
| `Vocabulary`      | Maps words to indices                 | Enables the LSTM to understand and predict words     |
| `Cosine Similarity` | Compares feature vectors             | Helps identify visually similar faces efficiently    |

---

## ✅ Features

- 🔁 End-to-end trainable image captioning model.
- 🧑‍🎤 Matches test faces to celebrity look-alikes.
- ✏️ Generates facial descriptions automatically.
- 🧩 Easily extendable to new datasets or feature sets.

---

## 📌 Future Work

- Add support for Transformer-based captioning.
- Integrate facial landmark detection for finer feature control.
- Build a web interface for real-time uploads and matches.

---

## 📷 Example Output (Coming Soon)

---

## 🤝 Contributions

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---
