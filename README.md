# Statistical-Learning
Internship projects and notebooks on Statistical Learning under the guidance of Prof. Neelesh S. Upadhye, IIT Madras.
# Statistical Learning Internship @ IIT Madras

Welcome to my official repository for the **Summer Internship (May–July 2025)** at the **Department of Mathematics, IIT Madras** under the guidance of **Prof. Neelesh S. Upadhye**.  
This internship focuses on practical and theoretical aspects of **Statistical Learning**, involving data modeling, evaluation, and optimization techniques using real-world datasets.

---

## About the Internship

- **Institute**: Indian Institute of Technology, Madras  
- **Mentor**: Prof. Neelesh S. Upadhye  
- **Duration**: 18 May 2025 – 18 July 2025  
- **Core Focus**: Statistical Modeling, Machine Learning, Applied Regression, Model Evaluation  
- **Base Environment**: [iitm-stats-learning/Conda_Env](https://github.com/iitm-stats-learning/Conda_Env)

---

## Tasks Completed

###  Week 1: Boston Housing Dataset
- Performed **Exploratory Data Analysis (EDA)**
- Built **OLS Regression Model** to predict housing prices
- Implemented **10-Fold Cross-Validation** for evaluation
- Visualized correlations, feature relationships, and residuals

 Notebook: `Rishabh_Shukla_Boston_HousingW1.ipynb`

---

###   Week 2: Bias-Variance & Model Tuning
- Studied and visualized **bias-variance tradeoff**
- Compared **underfitting vs overfitting models**
- Introduced **polynomial regression**
- Performed **model validation using RMSE and R²**

 Notebook: `Rishabh_Shukla_Heart_DiseaseW2.ipynb`

---

##  Environment Setup

This project uses the Conda environment defined here:  
🔗 [iitm-stats-learning/Conda_Env](https://github.com/iitm-stats-learning/Conda_Env)

### To Reproduce Locally:

```bash
git clone https://github.com/iitm-stats-learning/Conda_Env.git
cd Conda_Env
conda env create -f stats_learning.yml
conda activate stats-learning
jupyter notebook
