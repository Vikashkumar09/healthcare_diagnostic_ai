
#  Healthcare Diagnostic AI

An **AI-powered medical imaging diagnostic system** that detects **brain tumors from MRI scans** using deep learning.
This project demonstrates how Artificial Intelligence can assist healthcare professionals by providing **fast, accurate, and automated diagnostic insights** from medical images.

The system leverages **Convolutional Neural Networks (CNNs)** for image classification and provides an API interface for integrating the model into real-world healthcare applications.

---

#  Project Description

Healthcare Diagnostic AI is an **end-to-end deep learning pipeline** designed to automatically analyze MRI scans and detect the presence of brain tumors.

The system performs the following tasks:

• Accepts MRI images as input
• Preprocesses and normalizes medical images
• Runs inference using a trained deep learning model
• Predicts tumor presence with confidence score
• Returns results through a **FastAPI-based API**

This architecture enables **scalable deployment using Docker**, making the system production-ready.

---

# Features

✔ Automated **Brain Tumor Detection** using Deep Learning
✔ **CNN-based Image Classification Model**
✔ **FastAPI REST API** for inference
✔ **Dockerized Deployment** for reproducibility
✔ Modular **Training & Evaluation Pipeline**
✔ GPU-compatible architecture
✔ Clean and scalable project structure
✔  **cloud deployment (AWS )**

---

# System Architecture

```
            MRI Image
                │
                ▼
        Image Preprocessing
   (Resize, Normalize, Augment)
                │
                ▼
        Deep Learning Model
        (CNN / PyTorch Model)
                │
                ▼
        Model Inference Engine
                │
                ▼
          FastAPI Backend
                │
                ▼
           JSON Prediction
     (Tumor / No Tumor + Confidence)
```

---

# Tech Stack Used

### Programming

* Python 3.10

### Deep Learning

* PyTorch
* Transformers
* TorchVision
* NumPy
* Scikit-learn

### Computer Vision

* OpenCV
* CNN
* Pillow

### API Framework

* FastAPI
* Uvicorn

### Model Development

* Matplotlib
* HuggingFace Transformers (optional experiments)

### DevOps

* Docker
* Git
* GitHub

### Infrastructure

* Linux Containers
* WSL2
* GPU Runtime (optional)

---

# 📂 Project Structure

```
healthcare_diagnostic_ai
│
├── data
│   ├── raw
│   └── processed
│
├── models
│   └── brain_tumor_model.pth
│
├── training
│   ├── train.py
│   ├── evaluate.py
│   └── dataset.py
│
├── api
│   └── main.py
│
├── inference
│   └── predict.py
│
├── notebooks
│   └── exploration.ipynb
│
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Vikashkumar09/healthcare_diagnostic_ai.git
cd healthcare_diagnostic_ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🐳 Run with Docker

Build Docker image

```bash
docker build -t brain-tumor-ai .
```

Run container

```bash
docker run -p 8000:8000 brain-tumor-ai
```

---

# 🔬 Model Training

Train the CNN model:

```bash
python training/train.py
```

Evaluate the model:

```bash
python training/evaluate.py
```

---

# 🌐 Run API Server

```bash
uvicorn api.main:app --reload
```

API will run at:

```
http://localhost:8000
```

Swagger API Docs:

```
http://localhost:8000/docs
```

---

# 📊 Future Improvements

• Integrate **YOLO for tumor localization**
• Deploy using **Kubernetes**
• Add **Explainable AI (Grad-CAM)**
• Train with **larger medical datasets**
• Build **web dashboard for doctors**

---

# 👨‍💻 Author

**Vikash Kumar**

AI / Machine Learning Engineer
Focused on building **real-world AI systems in healthcare, education, and automation**.

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
🚀 Contribute to improve the system

## ☁️ AWS Deployment

This project is deployed on AWS using Docker and Amazon EC2.

### Steps

1. Launch an EC2 instance (Ubuntu 22.04 recommended)
2. Install Docker
3. Clone the repository
4. Build and run the Docker container

### EC2 Setup

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

### Clone Repository

```bash
git clone https://github.com/Vikashkumar09/healthcare_diagnostic_ai.git
cd healthcare_diagnostic_ai
```

### Build Docker Image

```bash
docker build -t brain-tumor-ai .
```

### Run Container

```bash
docker run -p 8000:8000 brain-tumor-ai
```

### Access API

```
http://EC2_PUBLIC_IP:8000/docs
```

This will launch the FastAPI documentation interface.

