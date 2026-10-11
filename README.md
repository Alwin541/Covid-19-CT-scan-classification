
# COVID-19 CT Scan Classification using MobileNetV3

This repository contains a PyTorch-based Deep Learning pipeline to classify preprocessed 3D CT scan slices into three distinct clinical categories:
1. **Normal CT scan (`nCT`)**
2. **COVID-19 positive CT scan (`pCT`)**
3. **Non-informative CT (`NiCT`)** (Lungs are not sufficiently visible to judge)

Despite a moderate class imbalance across the ~19,600 images in the dataset, the trained model achieves an outstanding test accuracy of **97.73%** and balanced precision/recall scores across all classes.

---

## 📊 Dataset & Class Distribution

The model is trained on the [Preprocessed CT Scans for COVID-19](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19) dataset. The distribution of the classes is as follows:

* **Normal CT scan (`nCT`)**: 9,979 images (~50.7%)
* **Non-informative CT (`NiCT`)**: 5,705 images (~29.0%)
* **COVID-19 positive (`pCT`)**: 4,001 images (~20.3%)

### Data Splits (Stratified Partition)
* **Train**: 13,778 images
* **Validation**: 2,951 images
* **Test**: 2,956 images

---

## ⚙️ Model Architecture

- **Base Backbone**: MobileNetV3-Large (Pretrained on ImageNet)
- **Classifier Head**:
  - Fully Connected (`Linear(960 -> 1280)`)
  - `ReLU` + `Dropout(p=0.2)`
  - Fully Connected (`Linear(1280 -> 3)`)
  - `LogSoftmax` output
- **Optimizer**: AdamW (Learning Rate: `1e-4`, Weight Decay: `1e-4`)
- **Loss Function**: Negative Log-Likelihood Loss (`NLLLoss`)

---

## 📈 Training and Evaluation Results

The network was trained for **25 epochs** on a single NVIDIA Tesla T4 GPU in Google Colab.

### Performance Metrics:

| Dataset | Loss | Accuracy |
| :--- | :---: | :---: |
| **Train** | 0.0882 | 96.85% |
| **Validation** | 0.0640 | 98.00% |
| **Test (Unseen)** | **0.0509** | **97.73%** |

### Class-Wise Metrics (Test Set Evaluation):

```
                                                       precision    recall  f1-score   support

Non-informative CT (lungs not visible enough to judge)       0.97      0.96      0.97       857
                                        Normal CT scan       0.98      0.98      0.98      1498
                             COVID-19 positive CT scan       0.98      0.98      0.98       601

                                              accuracy                           0.98      2956
                                             macro avg       0.98      0.98      0.98      2956
                                          weighted avg       0.98      0.98      0.98      2956
```

*Even with the minority COVID-19 positive class (`pCT`), the model maintains an **F1-score of 98%**, verifying that the training did not suffer from class-imbalance bias.*

---

## 🚀 How to Run & Replicate

### 1. Prerequisites
Ensure you have Python 3.8+ and PyTorch installed. Install dependencies:
```bash
pip install torch torchvision pandas numpy scikit-learn seaborn matplotlib pillow kagglehub
```

### 2. Download Dataset
```python
import kagglehub
path = kagglehub.dataset_download("azaemon/preprocessed-ct-scans-for-covid19")
```

### 3. Inference with Saved Checkpoint
```python
# Re-build and load the checkpoint
from your_code_module import load_checkpoint, predict_classes_names

model = load_checkpoint("outputs/checkpoints/mobilenet_v3_large.pth", device="cuda")
probs, class_names = predict_classes_names("path_to_ct_scan.jpg", model)
```
