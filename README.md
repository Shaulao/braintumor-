# 🧠 Brain Tumor Detection using Deep Learning

This is a deep learning-based project to detect brain tumors from MRI images using Convolutional Neural Networks (CNNs). The project aims to assist in early diagnosis of brain tumors, which is crucial for timely and effective medical intervention.

---

## 📌 Project Highlights

- Binary image classification: Tumor vs No Tumor  
- Built using CNNs with TensorFlow/Keras  
- High accuracy model trained on real MRI scans  
- Visualizations for predictions and training analysis  
- Easily extensible for segmentation/localization tasks  

---

## 👨‍🔬 Project Lead

- **Arjun Lakhanpal**  
  GitHub: [@arjunlakhanpall](https://github.com/arjunlakhanpall)

## 🤝 Contributors

- **Aditya Mishra**  
  Role: Data preprocessing, model evaluation, documentation, visualization

---

## 📁 Directory Structure

```
brain-tumor-detection/
│
├── dataset/
│   ├── yes/                 # MRI scans with tumors
│   └── no/                  # MRI scans without tumors
│
├── notebooks/               # Jupyter notebooks for exploration
│   └── EDA_and_Training.ipynb
│
├── models/                  # Saved model files
│
├── utils/                   # Utility functions (e.g., data loaders)
│
├── main.py                  # Main training and evaluation script
├── predict.py               # Script for predicting on new images
├── requirements.txt         # Python dependencies
└── README.md                # This file
```
![image](https://github.com/user-attachments/assets/8a0085e6-a091-4030-97cf-835614d29234)

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/arjunlakhanpall/brain-tumor-detection.git
cd brain-tumor-detection
```

2. **Install required libraries:**

```bash
pip install -r requirements.txt
```

---

## 🧠 Dataset Source

The dataset used is publicly available on Kaggle:

> **Brain Tumor MRI Dataset**  
> https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

The dataset consists of labeled MRI images categorized as **yes** (tumor) and **no** (no tumor).

---

## 🧮 Model Architecture

The CNN model includes:

- 3 Convolutional layers with ReLU + MaxPooling  
- Dropout layers for regularization  
- Flatten layer followed by dense fully connected layers  
- Final output layer with sigmoid activation for binary classification  

> **Optimizer**: Adam  
> **Loss Function**: Binary Crossentropy  
> **Metrics**: Accuracy, Precision, Recall

---
![image](https://github.com/user-attachments/assets/3e496749-f54a-42ff-a4ca-cb99a5bf7cc4)


## 🚀 How to Use

### Train the model:

```bash
python main.py
```

### Predict using a saved model:

```bash
python predict.py --image path_to_image.jpg
```

---

## 📊 Model Performance

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 95%+    |
| Precision | 94.8%   |
| Recall    | 96.2%   |
| F1-Score  | 95.5%   |

> *Note: Results may vary slightly depending on data split and training duration.*

---

## 📈 Visualizations

- Training vs Validation Accuracy & Loss  
- Confusion Matrix  
- Sample Predictions with Probabilities  

All available in the `EDA_and_Training.ipynb` notebook.

---

## 🔮 Future Enhancements

- Add tumor segmentation (e.g., using U-Net)  
- Deploy model using Streamlit/Flask  
- Build a mobile/web interface for real-time prediction  
- Expand to multiclass classification (tumor types)

---
![image](https://github.com/user-attachments/assets/a9f3af66-b488-4f2f-ab40-b07cf9a28e22)

## 📜 License

This project is open-source under the MIT License. Feel free to use, modify, and contribute.

---
![image](https://github.com/user-attachments/assets/a73a216f-3b37-4672-afe2-e88229504553)
![image](https://github.com/user-attachments/assets/b7ffec6c-935f-4f00-99ef-b9f057548f65)

## 🙌 Acknowledgements

- Kaggle Dataset by Navoneel Chakrabarty  
- TensorFlow and Keras Community  
- Guided by Arjun Lakhanpal with valuable contributions from Aditya Mishra

---

## 🔗 Connect

- Arjun Lakhanpal: [GitHub](https://github.com/arjunlakhanpall)  
- Aditya Mishra: *Coming soon...*
