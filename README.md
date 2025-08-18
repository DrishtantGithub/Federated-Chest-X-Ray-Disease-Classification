# 🩺 Federated Chest X-Ray Disease Classification  

## 📌 Overview  
This project explores **multi-label classification of chest diseases** using **deep learning** and **federated learning** techniques. Leveraging the **NIH ChestX-ray8 dataset (108k+ images, 32k+ patients)**, we benchmarked multiple state-of-the-art CNN architectures and enhanced performance using **federated client–server training**. To further improve robustness, we incorporated a **wavelet-based preprocessing pipeline** for noise reduction.  

---

## 🗂 Dataset  
- **NIH ChestX-ray8 dataset**  
- **108,948** frontal chest X-ray images  
- **32,717 unique patients**  
- **14 disease labels** (multi-label classification):  
  Hernia, Cardiomegaly, Emphysema, Effusion, Infiltration, Mass, Nodule, Atelectasis, Pneumothorax, Pleural Thickening, Pneumonia, Fibrosis, Edema, Consolidation  

---

## ⚙️ Data Processing  
- **Patient-level split** to avoid data leakage  
- **Data augmentation** + input normalization (mean 0, std 1)  
- **Grayscale → 3-channel conversion** for pretrained CNNs  
- **Image size**: 320 × 320 px  
- **Class imbalance handling** via weighted loss (w_pos & w_neg per class)  

---

## 🏗 Methodology  

### 🔹 Models Implemented  
- **Inception** (2014)  
- **ResNet** (2015)  
- **MobileNet** (2017)  
- **EfficientNet** (2019)  

### 🔹 Training Details  
- **Loss**: Weighted Mean Squared Error  
- **Optimizer**: Adam  
- **Learning rate scheduler** (adaptive to training loss)  
- **Metrics**: ROC AUC  

### 🔹 Federated Learning  
- Implemented using **ResNet-34** as the base client model  
- Simulated **multi-client setup** with server aggregation  
- Ensured no data leakage within splits  
- Clients trained independently → updates aggregated centrally  

### 🔹 Wavelet Preprocessing  
- Used **Haar Discrete Wavelet Transform (DWT)**  
- Focused on **LL (low-frequency) component** for denoising  
- Preserved structural patterns while reducing noise  
- Enhanced feature extraction before CNN training  

---

## 📊 Results  

### ROC AUC Comparison (Qualitative Order)  
**EfficientNet > ResNet > Inception > MobileNet**  

- **EfficientNet**: Strongest overall accuracy but resource-heavy  
- **ResNet**: Robust + easier to train (best trade-off)  
- **MobileNet**: Lightweight but less effective in high-resource setups  
- **Inception**: Efficient, but largely outperformed by newer models  

### Federated Learning  
- Improved generalization when compared to centralized training with limited subsets  
- Enabled **scalable decentralized training**  

### Wavelet Preprocessing  
- Noise reduction improved classification accuracy  
- Highlighted the **benefits of signal processing + deep learning hybrid approaches**  

