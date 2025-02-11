# Pokémon TCG Card Classifier  

**Project Type:** Computer Vision Model for Image Classification  
**Frameworks:** PyTorch, OpenCV

---

## 📌 Project Overview  

This was my final project for my Deep Learning class and consists of a **deep learning-based classification system** that can automatically identify the **type** of a Pokémon Trading Card Game (PTCG) card using **computer vision techniques**. It is based around two main components:  

1. **Classification Model:** A deep learning model trained on Pokémon card images to classify them by type (Fire, Water, Psychic).  
2. **Real-Time Detection System:** A script using OpenCV that captures live video, detects a Pokémon card, and overlays the predicted type in real-time.  

This model provides a quick way to recognize Pokémon card types using **transfer learning** with **ResNet18**.

---

## 📂 Dataset Source & Preprocessing  

### Dataset Source  
The dataset used in this project was obtained from **Hugging Face**:  
🔗 **[TheFusion21/PokemonCards Dataset](https://huggingface.co/datasets/TheFusion21/PokemonCards)**  

This dataset contains **13,139 Pokémon cards** from **1999 to 2022** in English, with high-resolution images and metadata such as `id`, `name`, `caption`, `hp`, and `set name`.

---

### Dataset Preprocessing  
The dataset required cleaning and preprocessing before training. The following steps were performed:

1. **Downloaded and assigned ID** to each card.
2. **Extracted the Pokémon type** from the `caption` field and created a new `type` column.
3. **Removed missing images** (51 cards removed, final dataset size: **13,087 cards**).
4. **Mapped Pokémon types to numeric indices** for model training:

   | Pokémon Type | Class Index |
   |-------------|------------|
   | Darkness    | 0          |
   | Colorless   | 1          |
   | Grass       | 2          |
   | Water       | 3          |
   | Metal       | 4          |
   | Psychic     | 5          |
   | Lightning   | 6          |
   | Dragon      | 7          |
   | Fire        | 8          |
   | Fighting    | 9          |
   | Fairy       | 10         |

5. **Split the dataset** into training, validation, and test sets (stratified split):

   | Dataset Split  | Size (Cards) |
   |---------------|-------------|
   | Training Set  | 9,160       |
   | Validation Set | 1,963       |
   | Test Set      | 1,964       |

---

## 🧠 Model Training (Classification Component)  

### Architecture
- **Base Model:** `ResNet18` from `torchvision.models`
- **Modification:** Final fully connected layer replaced to output **11 classes** (one per Pokémon type).
- **Preprocessing:** Resized images to **224×224**, normalized using ImageNet means and std.

### Hyperparameters
| Parameter      | Value |
|---------------|-------|
| Epochs        | 10    |
| Batch Size    | 32    |
| Learning Rate | 0.001 |
| Optimizer     | Adam  |
| Augmentations | Random crops, flips, rotations, color jitter |

---

### Training Results
- **Training Accuracy:** **94.8%**
- **Validation Accuracy:** **95.4%**
- **Test Accuracy:** **95.82%**
- **Loss:** **0.1286** (Final Test Set)

#### Training Notes
- Training took **~45 minutes** on an **RTX 3060 (CUDA)**.
- Accuracy was stable except for a dip on **epoch 4**, but it recovered.
- Stratified sampling helped address class imbalances (e.g., Dragon & Metal types had fewer examples).

---

## 🎥 Real-Time Detection System (Live Feed Component)  

### ⚠️ Limitations
This is a **raw OpenCV implementation**. It can:  
✅ Detect **one card at a time**.  
✅ Only classify **vertical cards** under **ideal lighting conditions**.  
⚠️ **Fails under poor lighting** or **angled cards**.  

---

### 🛠️ How It Works
1. **Video Capture:** Reads frames from a **live webcam/IP stream**.
2. **Edge Detection:** Uses OpenCV’s `cv2.Canny` to find contours.
3. **Card Detection:** Identifies the **largest rectangular contour** with **4 corners**.
4. **Classification:** Extracts and preprocesses the detected card, feeds it into **ResNet18**, and predicts its type.
5. **Overlay Display:** Draws a **bounding box** around the card and overlays **type name + icon**.

### 📈 Performance
- Best results under **cold natural light (10:00 - 11:30 AM)**.
- Worked best with **black backgrounds**.
- Successfully classified **90% of test cards** from **Paradox Rift (2023) & Obsidian Flames (2024)**.
- Struggled with **rotated cards** and **low-light environments**.

## 📜 References
- Dataset: TheFusion21/PokemonCards - Hugging Face
  https://huggingface.co/datasets/TheFusion21/PokemonCards
- OpenCV Docs: https://opencv.org
- PyTorch Docs: https://pytorch.org
- Related Projects:
  - Playing Card Classifier (Wijekoon, 2024)
    https://github.com/hiroonwijekoon/pytorch-cnn-playing-cards-classifier
  - Identifying Pokémon Cards (Peixoto, 2021)
    https://github.com/hugopeixoto/ptcg-detection

---

## 🔗 Additional Resources
📺 **Testing Video 1 (Early Attempt):**
https://youtu.be/_wima84K0NU  

📺 **Testing Video 2 (Latest Attempt):**
https://youtu.be/StDAPjqSBLo  

---
