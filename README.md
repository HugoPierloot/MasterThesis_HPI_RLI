# MasterThesis_HPI_RLI
# 🚀 Rocket Launch Failure Prediction

This repository contains the code and analysis for my Master Thesis:

**"Rocket Launch Failure Prediction Using Machine Learning"**

The project focuses on analyzing historical rocket launch data and building predictive models to identify failure patterns.

---

## 📂 Project Structure
MasterThesis_HPI_RLI/
│── main/ # Python scripts (analysis, modeling, etc.)
│── data/ # Dataset (ignored in Git)
│── figures/ # Generated visualizations
│── tables/ # Generated tables
│── requirements.txt
│── README.md

## 📊 Data Access

The dataset is **not included in this repository**.

### Why?

- File size constraints
- Potential licensing restrictions
- Best practices for Git repositories

The `data/` folder is therefore included in `.gitignore`.

---

## 📥 How to Use the Dataset

You must manually add the dataset to the following location: `data/Rocket_Launch_Industry_Dataset_Clean.xlsx`

### Required Excel structure

The file must contain the following sheets:

- `Launches`
- `Configs`
- `Families`
- `Companies`
- `Locations`

---

## ▶️ Running the Code

1. Clone the repository
2. Install dependencies : `pip install -r requirements.txt`
3. Run the data_analysis script : python main/data_analysis.py