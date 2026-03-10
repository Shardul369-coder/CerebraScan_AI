# 🧠 CerebraScan AI <br>

![Python](https://img.shields.io/badge/Python-3.10-blue)
![DVC](https://img.shields.io/badge/DVC-2.0+-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-yellow)
![AWS S3](https://img.shields.io/badge/AWS-S3-ff9900)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

### Reproducible Neuro-Imaging & Clinical Pipeline <br>

🚧 Status: In Active Development (Not Production Ready Yet)

---

## 🧾 Overview <br>

CerebraScan AI is an end-to-end multimodal MRI analysis system for **brain tumor segmentation**, **volumetric analysis**, **radiogenomic inference**, and **automated clinical reporting**, while integrating production-grade **MLOps**, **data versioning**, and **deployment tooling**.<br><br>

It combines:<br>
• **Deep Learning (U-Net & nnU-Net)**<br>
• **Radiogenomics**<br>
• **3D Visualization**<br>
• **Longitudinal Tracking**<br>
• **LLM-based Reporting**<br>
• **DVC + S3 Data Management**<br>
• **MLflow Experiment Tracking**<br>
• **Dockerized Deployment**<br><br>

to support neuro-oncology research and clinical workflows.

---

## 🎯 Clinical Approach & Research Objectives <br>

This system targets real-world neuro-oncology use-cases through the following objectives:<br><br>

1. **Volumetric 3D Tumor & Edema Analysis** using normalized T1, T1CE, T2, and FLAIR modalities.<br>
2. **Color-Coded Sub-Region Annotation** for edema, NET, ET, and background.<br>
3. **Intracranial Tumor Classification** (Glioma, Meningioma, Pituitary, Schwannomas, Medulloblastoma, Ependymoma).<br>
4. **2D→3D Fly-Through Visualization** with grayscale or RGB rendering.<br>
5. **Longitudinal Delta Tracking** for growth/shrinkage velocity and treatment response.<br>
6. **Surgical No-Go Zone Identification** to analyze anatomical adjacency and risk.<br>
7. **Biopsy Site Recommendation** via hotspot-based heatmaps for high-grade tissue sampling.<br>
8. **Radiogenomic Virtual Biopsy** predicting **IDH mutation** & **MGMT methylation**.<br>
9. **Automated LLM-Based Reporting** that consolidates all patient findings into structured radiology-style documentation.<br>

---

## 🧱 System Implementation <br>

### **1. Segmentation Backbone** <br>

Supported models:<br>
• **U-Net (TensorFlow/Keras)** — baseline 2D/3D segmentation<br>
• **nnU-Net (PyTorch)** — auto-configured SOTA clinical segmentation<br><br>

nnU-Net is used as the default segmentation engine due to its automated preprocessing, architecture tuning, and training pipeline.<br><br>

### **2. Classification** <br>

Post-segmentation, extracted radiomics + intensity features can be used for:<br>
• Tumor type classification<br>
• Low-grade vs high-grade prediction<br><br>

### **3. Radiogenomics** <br>

Using **TCGA-GBM**, the system aims to non-invasively predict:<br>
• **IDH mutation status**<br>
• **MGMT promoter methylation**<br><br>

### **4. Volumetric Analysis** <br>

3D reconstruction enables:<br>
• ET / NET / Edema volume extraction<br>
• Ratio & spread metrics<br>
• Treatment response quantification<br><br>

### **5. Visualization** <br>

Supports:<br>
• 2D slice overlays<br>
• 3D surface rendering<br>
• RGB segmentation overlays<br>
• Fly-through mode<br><br>

### **6. Delta Tracking** <br>

For multi-session scans of the same patient, the system computes:<br>
• Tumor growth rate<br>
• Edema evolution<br>
• Volumetric velocity maps<br>
• Clinical progression markers<br><br>

---

## 🌟 Feature Summary <br>

• Multimodal MRI segmentation<br>
• Clinical 3D volumetric mapping<br>
• Sub-region color coding<br>
• Tumor-type classification<br>
• Radiogenomic virtual biopsy<br>
• Biopsy hotspot recommendation<br>
• Surgical risk/no-go analysis<br>
• Longitudinal progression tracking<br>
• DICOM/NIfTI pipeline<br>
• MLOps-ready architecture<br>
• Automated structured reporting<br>

---

## 📂 Dataset & Modalities <br>

**BraTS 2023**<br>
• **Task:** Tumor sub-region segmentation<br>
• **Modalities:** T1, T1CE, T2, FLAIR<br>
• **Labels:** Edema, NET, ET<br><br>

**TCGA-GBM**<br>
• **Task:** Radiogenomics + classification<br>
• **Labels:** IDH, MGMT, survival metadata<br>

---

## 🤖 Current Model Support <br>

• **U-Net (Keras/TensorFlow)**<br>

---

# 🧰 Production & MLOps Layer (DVC + S3 + MLflow) <br>

CerebraScan AI includes a **production-grade data & experiment workflow** using:<br><br>

• **DVC** for dataset versioning<br>
• **AWS S3** for remote storage<br>
• **MLflow** for tracking experiments & metrics<br>
• **Docker** for deployment<br>
• **Git branching** for parallel experimentation<br><br>

This transforms the project from a notebook-based ML experiment into a **scalable MLOps pipeline** for research labs & hospitals.<br>

---

## 🧱 System Architecture <br>

Admin (CPU)<br>
│<br>
│ dvc push (write)<br>
▼<br>
AWS S3 Bucket<br>
▲<br>
│ dvc pull (read)<br>
│<br>
GPU Machines (Training/Inference)<br>
│<br>
└── git push (experiment branches)<br>
▼<br>
Git Remote (GitHub/GitLab)<br>

---

## 📦 Tech Stack <br>

| Component | Tool |
|---|---|
| Data Versioning | DVC |
| Remote Storage | S3 |
| Experiment Tracking | MLflow |
| Access Control | IAM |
| Model Training | GPU Machines |
| Formats | NIfTI / DICOM |
| Deployment | Docker |

---

## 🔐 Security Model <br>

• S3 stores sensitive dataset artifacts<br>
• DVC manages version references<br>
• Git stores only metadata & code<br>
• IAM restricts write-access to admin<br>
• If used in a team, teammates have read-only S3 access<br>

---

## 👤 Maintainer <br>


This project is maintained by **Shardul Salodkar**.<br>
