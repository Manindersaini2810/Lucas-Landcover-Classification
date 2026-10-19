# 🌍 LUCAS Land Cover Classification Using ResNet50

A deep learning project for land cover classification using Convolutional Neural Networks (CNNs), transfer learning, and the LUCAS dataset. This project improves validation accuracy from approximately **32% to over 80%** using ResNet50 and an optimized training pipeline.

## 📌 Project Overview

Land cover classification is an important computer vision task used in environmental monitoring, agricultural analysis, land-use planning, and geographical information systems.

This project explores deep learning techniques to classify land cover images into eight Level-1 categories using PyTorch. An initial ResNet18-based approach achieved approximately 32% validation accuracy. The pipeline was then improved using ResNet50 transfer learning, data augmentation, class balancing, and learning-rate scheduling, resulting in validation accuracy above 80%.

## 🎯 Objectives

- Develop a deep learning model for land cover image classification.
- Apply transfer learning using a pretrained ResNet50 architecture.
- Improve generalization through data augmentation.
- Address class imbalance using `WeightedRandomSampler`.
- Optimize model training using a learning-rate scheduler.
- Evaluate classification performance using validation metrics.

## 🗂️ Dataset

The project uses the LUCAS land cover dataset and its associated image data.

The target labels are organized into eight Level-1 land cover categories:

1. Artificial land
2. Cropland
3. Woodland
4. Shrubland
5. Grassland
6. Bare soil
7. Water areas
8. Wetland

**Input:** RGB land cover images.

**Preprocessing:** Images are resized to 224 × 224 pixels and normalized for the pretrained ResNet50 model.

The dataset is accessed separately and is not included in this repository. Please consult the dataset provider's documentation for access and licensing conditions.

## 🧠 Model Architecture

### ResNet50 with Transfer Learning

ResNet50 is a deep convolutional neural network that uses residual connections to facilitate the training of deeper models.

A pretrained ResNet50 model is adapted for the target land cover classification task by replacing its final classification layer with a layer corresponding to the eight target classes.

Transfer learning enables the model to reuse useful visual features learned during pretraining and adapt them to the target dataset.

## ⚙️ Methodology

### 1. Baseline Experiment

- Implemented an initial ResNet18-based classification pipeline.
- Experimented with segmentation preprocessing.
- Observed approximately 32% validation accuracy.
- Identified the need to improve the classification pipeline.

### 2. Transfer Learning

- Switched to pretrained ResNet50.
- Adapted the final classification layer for eight classes.
- Used pretrained visual representations to improve learning.

### 3. Data Augmentation

Applied training-time augmentation techniques to increase image variation and improve generalization:

- Random horizontal and vertical flips, where appropriate.
- Random rotations.
- Color jitter.

### 4. Class Imbalance Handling

Used PyTorch's `WeightedRandomSampler` to increase the sampling probability of underrepresented classes during training.

### 5. Learning-Rate Scheduling

Applied a learning-rate scheduler to adjust the learning rate during training and support more stable optimization.

### 6. Model Evaluation

Evaluated the trained model on a validation split and monitored classification accuracy. Further evaluation can include per-class precision, recall, F1-score, and a confusion matrix.

## 📊 Results

| Experiment | Validation Accuracy |
|---|---:|
| ResNet18 baseline with initial preprocessing | ~32% |
| ResNet50 with improved training pipeline | 80%+ |

The improved pipeline produced a substantial increase in validation accuracy.

*Note: These figures represent the reported validation results. Replace approximate values with exact measurements from your final experiment.*

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Computer Vision:** torchvision
- **Data Analysis:** NumPy, pandas
- **Evaluation:** scikit-learn
- **Visualization:** Matplotlib
- **Environment:** Kaggle Notebooks

## 📁 Repository Structure

```text
lucas-land-cover-classification/
│
├── notebooks/
│   └── lucas_resnet50_classification.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

*The structure above is a suggested layout. Include only files that actually exist in your repository.*

## 🚀 Installation and Usage

### Prerequisites

- Python 3.10 or a compatible Python version.
- pip package manager.
- Access to the LUCAS dataset.
- A GPU is recommended for faster training, but a CPU can be used.


### 1. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download or access the dataset through its authorized source. Configure the image directory and label paths to match the notebook's expected format.

### 4. Run the Notebook

Open the notebook in Jupyter or upload it to Kaggle, configure the dataset paths, and execute the cells in order.

## 📈 Future Improvements

- Compare ResNet50 with EfficientNet and other CNN architectures.
- Improve performance on underrepresented classes.
- Evaluate macro F1-score and class-wise precision and recall.
- Visualize errors using a confusion matrix.
- Test generalization on an independent dataset.
- Explore explainability methods such as Grad-CAM.
- Optimize the model for faster inference.

## 💡 Key Learnings

- Transfer learning can substantially improve performance on image classification tasks.
- Appropriate data augmentation helps improve model generalization.
- Class imbalance can affect overall and per-class performance.
- More complex preprocessing does not necessarily produce better results.
- Careful experimentation and validation are essential for improving deep learning models.


