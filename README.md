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
###  Week 3: Model Selection & Regularization
- Explored Model Selection : **Subset Selection**,**Stepwise Selection** , **Dimension Reduction**
- Explored regularization techniques: **Ridge**, **Lasso**, and **Elastic-Net**
- Conducted **feature engineering** and **data preprocessing** for model input optimization
- Implemented **Grid Search** for hyperparameter tuning
- Provided a markdown-based theoretical explanation of the bias-variance tradeoff
- Performed **cross-validation** to select optimal model parameters
  
 Notebook: `Rishabh_Shukla_W3.ipynb`

 ---
###  Additional Task: In-Depth Ridge and Lasso Regression's Algorithm
- Implemented Ridge Regression and Lasso Regression's algorithm from scratch using **normal equation** and **coordinate descent method**
- Examined the effect of regularization strength (λ) on coefficient shrinkage using **cross validation**
- Compared model performance for Ridge vs Lasso based on **evaluation metrics**
- Visualized coefficient paths and selected features under different regularization penalties
- Interpreted results in context of **model sparsity** and **overfitting control**

 Notebook: `Rishabh_Shukla_ridge_lasso_W4.ipynb`

 ---
###  Week 4: Tree-Based Methods (Regression Tree, Random Forest, XGBoost)
- Implemented **Regression Trees from scratch** including Recursive Binary Splitting and Cost-Complexity Pruning
- Used **cross-validation** to tune the cost-complexity parameter α and prune the tree effectively
- Visualized model structure and residual patterns with detailed diagnostic plots
- Built **Random Forest from scratch**, including bootstrap sampling and feature bagging
- Implemented **XGBoost from scratch** using sequential learning on residuals with shrinkage
- Compared all models on performance metrics (RSS, MSE, R²)
- Tuned hyperparameters using **manual grid search** for both Random Forest and XGBoost
- Provided theory markdowns inspired by **ISLR book** and aligned implementation with those concepts

 Notebook: `Rishabh_Shukla_non_linear_w4.ipynb`

 ---
## Reading Resources
- [An Introduction to Statistical Learning](https://www.statlearning.com/?utm_source=chatgpt.com)
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/?utm_source=chatgpt.com)
  
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

