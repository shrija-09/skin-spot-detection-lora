# Skin Spot Detection Using LoRA Fine-Tuned Models

This project presents a machine learning solution for classifying skin conditions as either benign (harmless) or malignant (cancerous) using dermatoscopic images. It utilizes Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique applied to a pre-trained Convolutional Neural Network (CNN). The project is implemented in Google Colab, providing a cloud-based, reproducible workflow for experimentation.

## Project Overview

The goal of this project is to support early detection of skin cancer by automatically classifying various types of skin lesions. The dataset includes labeled dermatoscopic images representing multiple skin conditions such as:

- Melanoma
- Basal Cell Carcinoma
- Nevus
- Eczema
- Vitiligo
- Warts
- Acne

The model architecture is built upon a standard CNN backbone, with LoRA applied to reduce training complexity and computational cost. This results in a model that is both efficient and accurate, making it suitable for deployment in mobile and edge environments.

## Key Features

- LoRA-based fine-tuning on a pre-trained CNN
- Multi-class classification of dermatological images
- Image preprocessing and augmentation techniques applied
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Trained and tested in Google Colab with GPU acceleration

## Tools and Technologies

- Python
- PyTorch
- Google Colab
- LoRA (Low-Rank Adaptation)
- NumPy, Pandas, Matplotlib
- scikit-learn
## Dataset

The dataset used in this project can be downloaded from the following Google Drive link:

[Download Dataset (ZIP)](https://drive.google.com/file/d/1t-XVbMo2gw5bi-Mf3LONoGz01AYClTPy/view?usp=drive_link)

It contains images of various skin conditions including:
- Acne
- Melanoma
- Psoriasis
- Actinic keratosis
- And more...

This dataset is used to train and evaluate the LoRA fine-tuned model for skin spot classification.

## Applications

This project highlights the potential of AI in dermatological diagnostics and can be extended for:

- Clinical decision support tools
- Mobile applications for skin anomaly screening
- Telemedicine platforms
- Research in parameter-efficient transfer learning

## Future Work

- Expand dataset to include more diverse skin conditions and demographics
- Deploy model on a mobile device using TensorFlow Lite or ONNX
- Integrate explainability techniques (e.g., Grad-CAM) to visualize model decisions

## Notebook

You can explore the full notebook here:
[skin_spot_detection.ipynb](./skin_spot_detection.ipynb)
