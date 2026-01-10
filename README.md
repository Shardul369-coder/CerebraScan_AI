# 🧠 CerebraScan AI — Reproducible Medical Imaging Pipeline                   (DVC + S3 + Git)

CerebraScan AI is a **reproducible, secure, team-friendly machine learning pipeline** designed for **MRI-based brain tumor preprocessing and segmentation**.

It features:

- **DVC** for data versioning & pipelines
- **AWS S3** for remote dataset storage
- **IAM** for secure access control
- **Git branching** for collaborative experiments
- **GPU teammates** for model training
- **Admin-only push rights** for protected medical data

---

## 🧱 System Architecture
Admin (CPU)
│
│ dvc push (write)
▼
AWS S3 Bucket
▲
│ dvc pull (read)
│
Team GPU Machines
│
└── git push (feature branches)
▼
Git Remote (GitHub/GitLab)



---

## 🔒 Access Control Model

| Role | DVC Push | DVC Pull | S3 Write | S3 Read | Git Push |
|---|---|---|---|---|---|
| Admin | ✔ | ✔ | ✔ | ✔ | ✔ |
| Teammate | ❌ | ✔ | ❌ | ✔ | ✔ |

---

## 📦 Tech Stack

| Component | Tool |
|---|---|
| Data Versioning | DVC |
| Remote Storage | AWS S3 |
| Access Control | IAM |
| Version Control | Git |
| Pipelines | dvc.yaml |
| Hyperparameters | params.yaml |
| Training Hardware | GPU Machines |
| Medical Formats | NIfTI (.nii) |

---

# 📖 Setup & Workflow Guide

## PHASE 1 — Admin Machine Setup (DVC + S3)

**Step 1 — Initialize DVC**
      dvc init

**Step 2 — Add Raw Data**
      dvc add raw_data

**Step 3 — Commit to Git**
      git add .
      git commit -m "Add raw data tracked by DVC"

**Step 4 — Create S3 Bucket**
      AWS Console → S3 → Create bucket
      Example: brain-tumor-data
      Region: ap-south-1 (recommended)

**Step 5 — Set DVC Remote**
      dvc remote add -d s3remote s3://brain-tumor-data


**Step 6 — Install AWS CLI**

**Step 7 — Configure AWS CLI**
      aws configure
          Enter:
            Access Key ID
            Secret Key
            Region (e.g., ap-south-1)
            Output: json
            
**Step 8 — IAM Setup for Admin**
      IAM → Users → Create user → dvc-admin
      Attach policy: AmazonS3FullAccess

**Step 9 — Push Data to S3**
      dvc push
      If upload interrupted run again: dvc push

## PHASE 2 — IAM Group Setup for Teammates

**Step 10 — Create IAM Group**
      IAM → User groups → Create → dvc-read-only

**Step 11 — Attach S3 Read-Only Policy**
      Attach:
          AmazonS3ReadOnlyAccess

## PHASE 3 — Add Teammates

**Step 12 — Create IAM Users**
      Example usernames:
          alice, bob, carol
      Disable console login, allow programmatic only.
          Add to group: dvc-read-only

**Step 13 — Generate Access Keys**
      IAM → User → Security credentials → Access keys → Create access key
          Select:
          Application running outside AWS
          Provide each teammate:
                Access Key ID
                Secret Access Key
                Region

## PHASE 4 — Teammate Setup (GPU Machines)

**Step 14 — Create Virtual Environment**
       Linux/Mac:
          python3 -m venv venv
          source venv/bin/activate
       Windows:
          python -m venv venv  
          venv\Scripts\activate

**Step 15 — Install Requirements**
        pip install -r requirements.txt


**Step 16 — Install AWS CLI**

**Step 17 — Configure AWS**
        aws configure

**Step 18 — Install Git & DVC**
        pip install dvc[s3]

**Step 19 — Clone Repo & Pull Data**
        git clone <repo-url>
        cd <repo-folder>  
        dvc pull

## PHASE 5 — Experiment Branching (GPU Teammates)

**Step 20 — Create Feature Branch**
        git checkout -b feature/<name-or-experiment>
        Examples:
          git checkout -b feature/john-unet-v2
          git checkout -b feature/sara-lr-3e-4


**Step 21 — Train on GPU Machine**

**Step 22 — Commit + Push Experiment**
         git add .
         git commit -m "Experiment: lr=3e-4"
         git push -u origin feature/john-lr-3e-4

## PHASE 6 — Sync Main Updates to Feature Branches

**Step 23 — If Admin updates main, teammates run:**
         git checkout main  
         git pull
         git checkout feature/<branch>
         git merge main     # or git rebase main
         Repeat Step 22 for pushing and commiting experiments if done on branch

## PHASE 7 — Admin Review of Teammate Branches

**Step 24 — Fetch All Branches**
         git fetch --all

**Step 25 — List Branches**
         git branch -a


**Step 26 — Checkout a Teammate Branch**
         git checkout feature/<name-or-experimet>
         Admin can now review code & validate updates.



## 🧩 Project Philosophy
**This project emphasizes:**
**reproducibility**
**data integrity**
**secure access control**
**collaboration via branches**
**separation of CPU vs GPU roles**
**centralized pipeline management**
**This matches real-world workflows used in research labs & medical imaging organizations**



## 🔐 Security Summary
**Dataset stored in S3**
**Teammates have read-only access**
**Only Admin runs dvc push**
**Data never travels through Git**
**S3 IAM prevents accidental deletion**
