# 🧠 CerebraScan AI — Reproducible Medical Imaging Pipeline                  

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
Admin (CPU)<br>
│<br>
│ dvc push (write)<br>
▼<br>
AWS S3 Bucket<br>
▲<br>
│ dvc pull (read)<br>
│<br>
Team GPU Machines<br>
│<br>
└── git push (feature branches)<br>
▼<br>
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

**Step 1 — Initialize DVC:**<br>
      dvc init

**Step 2 — Add Raw Data**<br>
      dvc add raw_data

**Step 3 — Commit to Git**<br>
      git add .<br>
      git commit -m "Add raw data tracked by DVC"

**Step 4 — Create S3 Bucket**<br>
      AWS Console → S3 → Create bucket<br>
      Example: brain-tumor-data<br>
      Region: ap-south-1 (recommended)

**Step 5 — Set DVC Remote**<br>
      dvc remote add -d s3remote s3://brain-tumor-data


**Step 6 — Install AWS CLI**

**Step 7 — Configure AWS CLI**<br>
      aws configure<br>
          Enter:<br>
            Access Key ID<br>
            Secret Key<br>
            Region (e.g., ap-south-1)<br>
            Output: json
            
**Step 8 — IAM Setup for Admin**<br>
      IAM → Users → Create user → dvc-admin<br>
      Attach policy: AmazonS3FullAccess

**Step 9 — Push Data to S3**<br>
      dvc push<br>
      If upload interrupted run again: dvc push<br>

## PHASE 2 — IAM Group Setup for Teammates

**Step 10 — Create IAM Group**<br>
      IAM → User groups → Create → dvc-read-only

**Step 11 — Attach S3 Read-Only Policy**<br>
      Attach:<br>
          AmazonS3ReadOnlyAccess

## PHASE 3 — Add Teammates

**Step 12 — Create IAM Users**<br>
      Example usernames:<br>
          alice, bob, carol<br>
      Disable console login, allow programmatic only.<br>
          Add to group: dvc-read-only

**Step 13 — Generate Access Keys**<br>
      IAM → User → Security credentials → Access keys → Create access key<br>
          Select:<br>
          Application running outside AWS<br>
          Provide each teammate:<br>
                Access Key ID<br>
                Secret Access Key<br>
                Region

## PHASE 4 — Teammate Setup (GPU Machines)

**Step 14 — Create Virtual Environment**<br>
       Linux/Mac:<br>
          python3 -m venv venv<br>
          source venv/bin/activate<br>
       Windows:<br>
          python -m venv venv<br>  
          venv\Scripts\activate

**Step 15 — Install Requirements**<br>
        pip install -r requirements.txt


**Step 16 — Install AWS CLI**

**Step 17 — Configure AWS**<br>
        aws configure

**Step 18 — Install Git & DVC**<br>
        pip install dvc[s3]

**Step 19 — Clone Repo & Pull Data**<br>
        git clone <repo-url><br>
        cd <repo-folder><br>  
        dvc pull

## PHASE 5 — Experiment Branching (GPU Teammates)

**Step 20 — Create Feature Branch**<br>
        git checkout -b feature/<name-or-experiment><br>
        Examples:<br>
          git checkout -b feature/john-unet-v2<br>
          git checkout -b feature/sara-lr-3e-4


**Step 21 — Train on GPU Machine**

**Step 22 — Commit + Push Experiment**<br>
         git add .<br>
         git commit -m "Experiment: lr=3e-4"<br>
         git push -u origin feature/john-lr-3e-4

## PHASE 6 — Sync Main Updates to Feature Branches

**Step 23 — If Admin updates main, teammates run:**<br>
         git checkout main<br>  
         git pull<br>
         git checkout feature/<branch><br>
         git merge main or git rebase main<br>
         Repeat Step 22 for pushing and commiting experiments if done on branch

## PHASE 7 — Admin Review of Teammate Branches

**Step 24 — Fetch All Branches**<br>
         git fetch --all

**Step 25 — List Branches**<br>
         git branch -a


**Step 26 — Checkout a Teammate Branch**<br>
         git checkout feature/<name-or-experimet><br>
         Admin can now review code & validate updates.



## 🧩 Project Philosophy
**This project emphasizes:** <br>
**reproducibility**<br>
**data integrity**<br>
**secure access control**<br>
**collaboration via branches**<br>
**separation of CPU vs GPU roles**<br>
**centralized pipeline management**<br>
**This matches real-world workflows used in research labs & medical imaging organizations**<br>



## 🔐 Security Summary
**Dataset stored in S3**<br>
**Teammates have read-only access**<br>
**Only Admin runs dvc push**<br>
**Data never travels through Git**<br>
**S3 IAM prevents accidental deletion**<br>
