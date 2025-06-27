# Import Important  libraries for All models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error
import random


# This function for Exploratory Data Analysis
def EDA_processing(df,response,expected_schema=None):
    # Drop unnecessary index column if present
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Display basic info
    print("\nDataset Overview")
    print(df.head())
    print(f"Shape: {df.shape}")
    print("\nColumn Info")
    print(df.info())
    print("\nSummary Statistics")
    print(df.describe())

    # Check and handle missing values
    print("\nMissing Values")
    print(df.isnull().sum())
    # Schema Validation (if provided)
    if expected_schema:
        print("\n Schema Validation:")
        for col, expected_type in expected_schema.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if actual_type != expected_type:
                    print(f"Type mismatch in '{col}': Expected {expected_type}, Got {actual_type}")
            else:
                print(f"Missing column in data: {col}")
    # Fill numeric columns with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Fill categorical columns with mode
    for col in df.select_dtypes(include='object'):
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    # Confirm no missing values remain
    print("\nAfter Handling Missing Values")
    print(df.isnull().sum())

    # Encode categorical variables
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Convert boolean columns to integers
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Correlation Matrix
    print("\nPearson Correlation Matrix:")
    correlation_matrix = df.corr(numeric_only=True)

    # Plot heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix ", fontsize=16)
    plt.show()

    # Correlation with target variable (response)
    print("\nCorrelation with Target (response)")
    Salary_corr = correlation_matrix[response].sort_values(ascending=False)
    print(Salary_corr)

    # Separate response and predictor variables
    responses = response
    predictors = [col for col in df.columns if col != response]
    print(f"\nResponse Variable: {responses}")
    print(f"Predictor Variables ({len(predictors)}): {predictors}")
    return df
# These are helper function
# All Helper Functions for Metric Evaluation and All Regression Tree
def compute_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def compute_mse(rss,n):
    return rss/n

def r2_score(y_true, y_pred):
    RSS = np.sum((y_true - y_pred) ** 2)
    TSS = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - RSS / TSS

def adjusted_r2_score(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

## Compute the prediction of tree
def predict_tree(tree, row):
    if tree["type"] == "leaf":
        return tree["prediction"]
    if row[tree["feature"]] < tree["split_val"]:
        return predict_tree(tree["left"], row)
    else:
        return predict_tree(tree["right"], row)
    
# Calculate Prediction of tree using numpy so that its evaluation more faster
def batch_predict(tree, data):
    return np.array([predict_tree(tree, row) for _, row in data.iterrows()])
# Compute Tree RSS
def compute_tree_rss(tree, data):
    preds = batch_predict(tree, data)
    return compute_rss(data["response"].values, preds)
# Count the tree leaves
def count_leaves(tree):
    if tree["type"] == "leaf":
        return 1
    return count_leaves(tree["left"]) + count_leaves(tree["right"])
# Compute tree depth
def get_tree_depth(tree):
    if tree["type"] == "leaf":
        return 0
    return 1 + max(get_tree_depth(tree["left"]), get_tree_depth(tree["right"]))

# Count internal nodes
def count_internal_nodes(tree):
    if tree["type"] == "leaf":
        return 0
    return 1 + count_internal_nodes(tree["left"]) + count_internal_nodes(tree["right"])

# Compute Cost Complexity of the Tree
def compute_cost_complexity(tree, alpha, data):
    rss = compute_tree_rss(tree, data)
    leaves = count_leaves(tree)
    return rss + alpha * leaves

# This is the Plotting Function
# Helper Function for Plotting 
def plot_predictions_vs_actuals(y_true, y_pred, title="Predicted vs Actual Values"):
    plt.figure(figsize=(14, 12))
    sns.scatterplot(x=y_true, y=y_pred, color='blue', s=25, edgecolor='black')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal Prediction')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# This is the Complete Linear Regression Function 
def linear_regression_fit(x,y) :
    # Forming X Matrix
    X=np.array(x)
    X=np.column_stack((np.ones((X.shape[0], 1)), X))
    # Use Normal Equation to fit the model 
    b=np.linalg.pinv(X.T@X)@X.T@y
    return b
def linear_regression_predict(x,b):
    X=np.array(x)
    X=np.column_stack((np.ones((X.shape[0], 1)), X))
    y_pred=X@b
    return y_pred

def evaluate_linear_regression(x,y,predictors):
    # Train/Test Split for the better modal evaluation 
    train_x, test_x ,train_y,test_y= train_test_split(x,y, test_size=0.2, random_state=42)
    b=linear_regression_fit(train_x,train_y)
    y_train_pred=linear_regression_predict(train_x,b)
    y_test_pred = linear_regression_predict(test_x,b)
    
    print("Estimated Coefficients :")
    for i in range(len(b)):
        if i==0:
            print(f"Intercept : {b[i]:.4f}")
        else :
            print(f"Coefficient for {predictors[i-1]} : {b[i]:.4f}")
    # Plotting Between Test Actual vs Test Predicted
    plot_predictions_vs_actuals(test_y.values,y_test_pred,title="Linear Regression (Predicted vs Actual)")
    # Determine number of predictors
    if len(x.shape) == 1:
        p = 1
    else:
        p = x.shape[1]

    # Metric Evaluation
    train_rss=compute_rss(train_y,y_train_pred)
    train_mse=compute_mse(train_rss,len(train_y))
    test_rss=compute_rss(test_y,y_test_pred)
    test_mse=compute_mse(test_rss,len(test_y))
    train_r2=r2_score(train_y,y_train_pred)
    test_r2=r2_score(test_y,y_test_pred)
    train_adj_r2=adjusted_r2_score(train_y,y_train_pred,p)
    test_adj_r2=adjusted_r2_score(test_y,y_test_pred,p)
    # Print All Metric Evaluation
    print(f"Train RSS : {train_rss :.4f}")
    print(f"Test RSS :{test_rss :.4f}")
    print(f"Train MSE :{train_mse :.4f}")
    print(f"Test MSE :{test_mse :.4f}")
    print(f"Train R^2 :{train_r2 :.2f}")
    print(f"Test R^2 :{test_r2 :.2f}")
    print(f"Train Adjusted R^2 :{train_adj_r2 :.2f}")
    print(f"Test Adjusted R^2 :{test_adj_r2 :.2f}")

# This is the Complete function for Ridge Regression 
# Making a Function Which Calculate The Ridge Coefficients with lambda values
def ridge_beta(X, y, lam):
    n_features = X.shape[1]
    I = np.eye(n_features)
    XT_X = X.T @ X
    XT_y = X.T @ y
    beta_ridge = np.linalg.solve(XT_X + lam * I, XT_y)  # Better than inv()
    return beta_ridge

# Finding Best Lambda Using Grid Search Using k Fold
def best_lambda(x,y,lambda_values=np.logspace(-5,5,50),k=10):
    avg_cv_errors=[]
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for lam in lambda_values:
        cv_errors = []
        for train_idx, val_idx in kf.split(x):
            X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]

            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


            beta_ridge = ridge_beta(X_train, y_train, lam)
            y_pred = X_val @ beta_ridge
            mse = mean_squared_error(y_val, y_pred)
            cv_errors.append(mse)
        avg_cv_errors.append(np.mean(cv_errors))
    # Select best lambda
    best_lambda = lambda_values[np.argmin(avg_cv_errors)]
    return best_lambda

def predict_ridge(x,b):
    X=np.column_stack((np.ones(x.shape[0]), x))
    y_pred=X@b
    return y_pred

# Evaluate Ridge Regression
def evaluate_ridge_regression(x,y,predictors):
    train_x, test_x ,train_y,test_y= train_test_split(x,y, test_size=0.2, random_state=42)
    # Standardise the train Data
    x_mean=np.mean(train_x,axis=0)
    x_std=np.std(train_x,axis=0)
    x_standardized = (train_x - x_mean) / x_std
    best_lam=best_lambda(x_standardized,train_y)
    final_beta=ridge_beta(x_standardized,train_y,best_lam)

    # Rescale the beta coefficietns
    final_beta_ridge=final_beta/x_std
    # Compute intercept (β₀) separately (from original data)
    beta_0 = np.mean(train_y) - np.dot(final_beta_ridge, x_mean)

    # Combine intercept and coefficients
    beta_full = np.insert(final_beta_ridge, 0, beta_0)
    y_train_pred=predict_ridge(train_x,beta_full)
    y_test_pred=predict_ridge(test_x,beta_full)

    # Display the coefficients
    print("Ridge Regression Coefficients:")
    for i, coef in enumerate(beta_full):
        if i == 0:
            print(f"Intercept (β₀): {coef:.4f}")
        else:
            print(f"Coefficient for {predictors[i-1]}: {coef:.4f}")
    # Plotting Between Test Actual vs Test Predicted

    plot_predictions_vs_actuals(test_y,y_test_pred,title="Ridge Regression (Predicted vs Actual)")
    # Determine number of predictors
    if len(x.shape) == 1:
        p = 1
    else:
        p = x.shape[1]

    # Metric Evaluation
    train_rss=compute_rss(train_y,y_train_pred)
    train_mse=compute_mse(train_rss,len(train_y))
    test_rss=compute_rss(test_y,y_test_pred)
    test_mse=compute_mse(test_rss,len(test_y))
    train_r2=r2_score(train_y,y_train_pred)
    test_r2=r2_score(test_y,y_test_pred)
    train_adj_r2=adjusted_r2_score(train_y,y_train_pred,p)
    test_adj_r2=adjusted_r2_score(test_y,y_test_pred,p)
    # Print All Metric Evaluation
    print(f"Train RSS : {train_rss :.4f}")
    print(f"Test RSS :{test_rss :.4f}")
    print(f"Train MSE :{train_mse :.4f}")
    print(f"Test MSE :{test_mse :.4f}")
    print(f"Train R^2 :{train_r2 :.2f}")
    print(f"Test R^2 :{test_r2 :.2f}")
    print(f"Train Adjusted R^2 :{train_adj_r2 :.2f}")
    print(f"Test Adjusted R^2 :{test_adj_r2 :.2f}")

# This is the Complete function for Lasso Regression 

# Now i am going to build lasso regression 
def soft_threshold(rho, lam):
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

def lasso_coordinate_descent(X, y, lam, tol=1e-4, max_iter=1000):
    n=X.shape[0]
    p = X.shape[1]
    beta = np.zeros(p)
    
    # Coordinate Descent Algorithm
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            X_j = X.iloc[:, j].values
            y_pred = X @ beta
            residual = y - y_pred + beta[j] * X_j # partial residual
            rho_j = np.dot(X_j, residual)
            beta[j] = soft_threshold(rho_j / n, lam) / (np.dot(X_j, X_j) / n)
        if np.sum(np.abs(beta - beta_old)) < tol:
            break
    return beta

def best_lambda_lasso(x,y,lambda_values=np.logspace(-5,5,50),k=10):
    avg_cv_errors=[]
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for lam in lambda_values:
        cv_errors = []
        for train_idx, val_idx in kf.split(x):
            X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]

            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


            beta_lasso = lasso_coordinate_descent(X_train, y_train, lam)
            y_pred = X_val @ beta_lasso
            mse = mean_squared_error(y_val, y_pred)
            cv_errors.append(mse)
        avg_cv_errors.append(np.mean(cv_errors))
    # Select best lambda
    best_lambda = lambda_values[np.argmin(avg_cv_errors)]
    return best_lambda
# Making a Function for Predictions
def predict_lasso(x,b):
    X=np.column_stack((np.ones(x.shape[0]), x))
    y_pred=X@b
    return y_pred
# Evaluate Lasso Regression
def evaluate_lasso_regression(x,y,predictors):
    train_x, test_x ,train_y,test_y= train_test_split(x,y, test_size=0.2, random_state=42)
    # Standardise and Centre the train Data
    x_mean=np.mean(train_x,axis=0)
    x_std=np.std(train_x,axis=0)
    x_standardized = (train_x - x_mean) / x_std
    y_mean=np.mean(train_y)
    y_centre=train_y-y_mean
    best_lam=best_lambda_lasso(x_standardized,y_centre)
    final_beta=lasso_coordinate_descent(x_standardized,y_centre,best_lam)

    # Rescale the beta coefficietns
    final_beta_lasso=final_beta/x_std
    # Compute intercept (β₀) separately (from original data)
    beta_0 = np.mean(train_y) - np.dot(final_beta_lasso, x_mean)

    # Combine intercept and coefficients
    beta_full = np.insert(final_beta_lasso, 0, beta_0)
    y_train_pred=predict_lasso(train_x,beta_full)
    y_test_pred=predict_lasso(test_x,beta_full)

    # Display the coefficients
    print("Lasso Regression Coefficients:")
    for i, coef in enumerate(beta_full):
        if i == 0:
            print(f"Intercept (β₀): {coef:.4f}")
        else:
            print(f"Coefficient for {predictors[i-1]}: {coef:.4f}")
    # Plotting Between Test Actual vs Test Predicted

    plot_predictions_vs_actuals(test_y,y_test_pred,title="Lasso Regression (Predicted vs Actual)")
    if len(x.shape) == 1:
        p = 1
    else:
        p = x.shape[1]
    # Metric Evaluation
    train_rss=compute_rss(train_y,y_train_pred)
    train_mse=compute_mse(train_rss,len(train_y))
    test_rss=compute_rss(test_y,y_test_pred)
    test_mse=compute_mse(test_rss,len(test_y))
    train_r2=r2_score(train_y,y_train_pred)
    test_r2=r2_score(test_y,y_test_pred)
    train_adj_r2=adjusted_r2_score(train_y,y_train_pred,p)
    test_adj_r2=adjusted_r2_score(test_y,y_test_pred,p)
    # Print All Metric Evaluation
    print(f"Train RSS : {train_rss :.4f}")
    print(f"Test RSS :{test_rss :.4f}")
    print(f"Train MSE :{train_mse :.4f}")
    print(f"Test MSE :{test_mse :.4f}")
    print(f"Train R^2 :{train_r2 :.2f}")
    print(f"Test R^2 :{test_r2 :.2f}")
    print(f"Train Adjusted R^2 :{train_adj_r2 :.2f}")
    print(f"Test Adjusted R^2 :{test_adj_r2 :.2f}")


# This is the Complete function for Regression Tress 
# All Main  Function for Regression Tree
# Now i am going to apply greedy Recursive Binary Split Approach to get full tree 
# This function heps to best split a region this is a helper function we use it in main grow tree function


def find_best_split(data, predictors):
    best_rss = float("inf") # Assign with infinite 
    best_split = None 
    y=data["response"].values
    for feature in predictors: # Checking for all predictors and there best cutpoints
        feature_values = np.sort(np.unique(data[feature].values))
        if len(feature_values) > 1:
            split_candidates = (feature_values[:-1] + feature_values[1:]) / 2
        else:
            continue  # Skip feature with no variation
        if len(split_candidates) > 100:
            split_candidates = np.random.choice(split_candidates, 100, replace=False)
            split_candidates = np.sort(split_candidates)  # Re-sort after sampling
        for s in split_candidates:
            left = data[feature].values< s # select that data where this condition is true
            right = ~left
            if np.sum(left) == 0 or np.sum(right) == 0: # Checking the emptyness of region
                continue
            y_left=y[left]
            y_right=y[right]
            y_left_pred = y_left.mean()
            y_right_pred = y_right.mean()
            rss = compute_rss(y_left, y_left_pred) + compute_rss(y_right, y_right_pred)
            if rss < best_rss:
                best_rss = rss
                best_split = { # Dictionary to store the information 
                    "feature": feature,
                    "split_val": s,
                    "rss": best_rss,
                    "left_mean": y_left_pred,
                    "right_mean": y_right_pred
                }
    return best_split

# This is the main function which calculate the complete grow tree
def grow_tree(data, predictors, min_samples_split=20, depth=0,max_depth=5):
    # Stopping condition
    if len(data) < min_samples_split or depth>=max_depth :
        return {
            "type": "leaf",
            "prediction": data["response"].mean(),
            "samples": len(data),
            "depth": depth
        }

    # Find best split
    best_split = find_best_split(data, predictors)
    if not best_split: # if no best split is possible 
        return {
            "type": "leaf",
            "prediction": data["response"].mean(),
            "samples": len(data),
            "depth": depth
        }
    
    
    # Recursively grow left and right branches
    left_data = data[data[best_split["feature"]] < best_split["split_val"]]
    right_data = data[data[best_split["feature"]] >= best_split["split_val"]]
    return {
        "type": "node",
        "feature": best_split["feature"],
        "split_val": best_split["split_val"],
        "left": grow_tree(left_data, predictors, min_samples_split, depth + 1),
        "right":grow_tree(right_data, predictors, min_samples_split, depth + 1),
        "depth": depth
    }
# Now make a function which prune the cost complexity for give some penality to reduce the varience
# Prune using cost complexity pruning for each alpha 
def prune_tree(tree, data, alpha):
    # Base case: if the node is already a leaf, return as-is
    if tree["type"] == "leaf":
        return tree

    # Split the data according to current node's condition
    left_data = data[data[tree["feature"]] < tree["split_val"]]
    right_data = data[data[tree["feature"]] >= tree["split_val"]]

    # Recursively prune left and right subtrees
    tree["left"] = prune_tree(tree["left"], left_data, alpha)
    tree["right"] = prune_tree(tree["right"], right_data, alpha)

    # Evaluate cost of current subtree
    subtree_rss = compute_tree_rss(tree, data)
    subtree_leaves = count_leaves(tree)
    subtree_cost = subtree_rss + alpha * subtree_leaves

    # Evaluate cost of pruning this node into a single leaf
    leaf_prediction = data["response"].mean()
    leaf_rss = compute_rss(data["response"].values, np.full(len(data), leaf_prediction))
    leaf_cost = leaf_rss + alpha * 1  # 1 leaf node

    # Decide whether to prune this node
    if leaf_cost <= subtree_cost:
        return {
            "type": "leaf",
            "prediction": leaf_prediction,
            "samples": len(data),
            "depth": tree["depth"]
        }
    else:
        return tree

# Evaluate the Tree
def evaluate_tree(tree, val_data):
    
    # Predict using the tree
    y_true = val_data["response"].values
    y_pred = batch_predict(tree, val_data)

    # Compute MSE
    rss = compute_rss(y_true, y_pred)
    mse = compute_mse(rss, len(val_data))
    
    return mse

# Cross-validation for each alpha to find best alpha

def cross_validate_pruning(data, predictors, alphas, k_folds=5, min_samples_split=20, max_depth=5):
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    alpha_errors = {alpha: [] for alpha in alphas}
    
    for train_index, val_index in kf.split(data):
        train_data = data.iloc[train_index].reset_index(drop=True)
        val_data = data.iloc[val_index].reset_index(drop=True)
        
        # Build full tree on training data
        full_tree = grow_tree(train_data, predictors, min_samples_split=min_samples_split, max_depth=max_depth)
        
        # Prune and evaluate for each alpha
        for alpha in alphas:
            pruned_tree = prune_tree(full_tree, train_data, alpha)
            mse = evaluate_tree(pruned_tree, val_data)
            alpha_errors[alpha].append(mse)
    
    # Average the errors
    alpha_mse_list = [(alpha, np.mean(mse_list)) for alpha, mse_list in alpha_errors.items()]
    
    # Select best alpha
    best_alpha = min(alpha_mse_list, key=lambda x: x[1])[0]
    
    return best_alpha, alpha_mse_list

# Function to get best subtree after pruning on best alpha
def final_pruned_tree(df, predictors, alpha, min_samples_split=20, max_depth=5):
  
    # Build full tree
    full_tree = grow_tree(df, predictors, min_samples_split=min_samples_split, max_depth=max_depth)

    # Prune it using selected alpha
    pruned_tree = prune_tree(full_tree, df, alpha)
    
    return pruned_tree


def evaluate_regression_tree(data, predictors, alpha_values=np.linspace(0, 10, 21), test_size=0.2, random_state=42, min_samples_split=20,max_depth=8):
    # Train/Test Split
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)

    # Select best alpha using CV on training data
    best_alpha, alpha_errors = cross_validate_pruning(train_df, predictors, alpha_values, k_folds=5, min_samples_split=min_samples_split,max_depth=max_depth)

    # Train best-pruned tree on training data and get best tree 
    best_tree = final_pruned_tree(train_df, predictors, best_alpha, min_samples_split=min_samples_split,max_depth=max_depth)

    # Step 4: Evaluation
    train_rss = compute_tree_rss(best_tree, train_df)
    test_rss = compute_tree_rss(best_tree, test_df)

    train_mse = train_rss / len(train_df)
    test_mse = test_rss / len(test_df)

    tree_depth = get_tree_depth(best_tree)
    n_leaves = count_leaves(best_tree)
    nodes=count_internal_nodes(best_tree)
    p=data.shape[1]-1
    y_test = test_df["response"].values
    test_predictions = test_df.apply(lambda row: predict_tree(best_tree, row), axis=1)
    y_test_pred=test_predictions.values
    test_r2 = r2_score(y_test, y_test_pred)
    # Calculating Train R_2 also 
    y_train = train_df["response"].values
    y_train_pred = train_df.apply(lambda row: predict_tree(best_tree, row), axis=1).values
    train_r2=r2_score(y_train,y_train_pred)
    train_adj_r2=adjusted_r2_score(y_train,y_train_pred,p)
    test_adj_r2=adjusted_r2_score(y_test,y_test_pred,p)
    # Print the results
    print("Final Regression Tree Evaluation")
    print(f"Best alpha (from CV): {best_alpha}")
    print(f"Tree Depth: {tree_depth}")
    print(f"Internal Nodes: {nodes}")
    print(f"Number of Leaves: {n_leaves}")
    print(f"Train RSS: {train_rss:.2f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test RSS: {test_rss:.2f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Train R_2 Score: {train_r2:.4f}")
    print(f"Test R_2 Score: {test_r2:.4f}")
    print(f"Train Adjusted R^2 :{train_adj_r2 :.2f}")
    print(f"Test Adjusted R^2 :{test_adj_r2 :.2f}")
    plot_predictions_vs_actuals(y_test,y_test_pred,title="Regression Tree (Predicted vs Actual)")

# This is The Complete Function for Random Forest
# Now i am going to build Randoom Forest
# Bootstrap Sampling the data
def bootstrap_sample(data):
    indices = np.random.randint(0, len(data), len(data))
    return data.iloc[indices].reset_index(drop=True)

# Train Random Forest With Feature Subsampling and Bootstrap
def train_random_forest(data,predictors, n_trees=10, max_depth=3, min_samples_split=2, max_features=None):
    np.random.seed(42)
    random.seed(42)

    forest = []
    for i in range(n_trees):
        sample = bootstrap_sample(data)
        max_features = min(max_features if max_features else int(np.sqrt(len(predictors))), len(predictors))
        subset_features = random.sample(predictors, max_features)
        tree = grow_tree(sample, subset_features, max_depth=max_depth, min_samples_split=min_samples_split)
        forest.append((tree, subset_features))
    return forest

# Make Predictions From all Trees in Forest with Forest

def predict_forest(forest, data):
    preds = []
    for tree, features in forest:
        pred = data.apply(lambda row: predict_tree(tree, row), axis=1)
        preds.append(pred)
    preds = np.array(preds)
    return np.mean(preds, axis=0)

def evaluate_random_forest(data,predictors,n_tress=10,max_depth=5,max_features=4):
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    rf_model = train_random_forest(train_df,predictors, n_trees=n_tress, max_depth=max_depth, max_features=max_features)
    p=data.shape[1]-1
    train_y=train_df["response"].values
    train_preds = predict_forest(rf_model, train_df)
    test_preds = predict_forest(rf_model, test_df)
    test_y=test_df["response"]
    train_rss=compute_rss(train_df['response'],train_preds)
    train_mse=train_rss/len(train_df)
    test_rss=compute_rss(test_df['response'],test_preds)
    test_mse=test_rss/len(test_df)
    plot_predictions_vs_actuals(test_y,test_preds,title="Random Forest (Predicted vs Actual)")
    train_r2 = r2_score(train_df["response"].values, train_preds)
    test_r2 = r2_score(test_df["response"].values, test_preds)
    train_adj_r2=adjusted_r2_score(train_y,train_preds,p)
    test_adj_r2=adjusted_r2_score(test_y,test_preds,p)

    print("Random Forest Metric Evaluation")
    print(f"Train_RSS :{train_rss:.4f}")
    print(f"Test_RSS :{test_rss :.4f}")
    print(f"Train MSE :{train_mse :.4f}")
    print(f"Test MSE :{test_mse :.4f}")
    print(f"Train R^2: {train_r2:.4f}")
    print(f"Test R^2: {test_r2:.4f}")
    print(f"Train Adjusted R^2 :{train_adj_r2 :.2f}")
    print(f"Test Adjusted R^2 :{test_adj_r2 :.2f}")

# This is The Complete Function for XGBoost 
# This Function Creaate all tress in Boosting 
def xgboost_fit(data, predictors, response_col, n_estimators=100, learning_rate=0.1, max_depth=2, min_samples_split=2):
    models = []       # Stores all fitted trees
    residuals=data[response_col]
    for i in range(n_estimators):
        # Create new dataset with residuals as the new target
        training_data = data[predictors].copy()
        training_data["response"] = residuals ## add a fake columns for predicting a tree 

        # Fit a shallow regression tree to residuals
        tree = grow_tree(training_data,predictors, min_samples_split=min_samples_split, depth=0, max_depth=max_depth)
        
        # Predict residuals
        pred = training_data.apply(lambda row: predict_tree(tree, row), axis=1)
        
        # Update the residuals
        residuals = residuals- learning_rate * pred
        models.append(tree)
    return models

# This function predict the final result of model
def xgboost_predict(models, data, learning_rate=0.1):
    total_pred = np.zeros(len(data))
    for tree in models:
        pred = data.apply(lambda row: predict_tree(tree, row), axis=1)
        total_pred += learning_rate * pred
    return total_pred

def evaluate_XGBoost(data,predictors,n_estimators=100,max_depth=2):
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    response_col="response"
    # Train XGBoost model
    xgb_model = xgboost_fit(train_df, predictors,response_col, n_estimators=n_estimators, learning_rate=0.1, max_depth=max_depth)
    p=data.shape[1]-1
    # Predict
    train_preds = xgboost_predict(xgb_model, train_df, learning_rate=0.1)
    test_preds_xg = xgboost_predict(xgb_model, test_df, learning_rate=0.1)
    y_true=test_df["response"]
    train_rss=compute_rss(train_df['response'],train_preds)
    train_mse=train_rss/len(train_df)
    test_rss=compute_rss(test_df['response'],test_preds_xg)
    test_mse=test_rss/len(test_df)
    train_adj_r2=adjusted_r2_score(train_df["response"],train_preds,p)
    test_adj_r2=adjusted_r2_score(y_true,test_preds_xg,p)
    plot_predictions_vs_actuals(y_true,test_preds_xg,title="XGBoost (Predicted vs Actual)")
    # Evaluate
    print("XGBoost Metric Evaluation ")
    print(f"Train_RSS :{train_rss:.4f}")
    print(f"Test_RSS :{test_rss :.4f}")
    print(f"Train MSE :{train_mse :.4f}")
    print(f"Test MSE :{test_mse :.4f}")
    print(f"Train R^2: {r2_score(train_df['response'], train_preds):.4f}")
    print(f"Test R^2: {r2_score(test_df['response'], test_preds_xg):.4f}")
    print(f"Train Adjusted R^2 :{train_adj_r2 :.2f}")
    print(f"Test Adjusted R^2 :{test_adj_r2 :.2f}")

# This function is mix concept of xgboost technique and linear Regression
def xgboost_linear_regression_fit(x,y,n_estimator,learning_rate):
    models=[]
    residual=y
    for i in range(n_estimator):
        b=linear_regression_fit(x,residual)
        pred=linear_regression_predict(x,b)
        residual=residual-learning_rate*pred
        models.append(b)
    return models

def xgboost_linear_regression_predict(models,x,learning_rate):
    final_b=np.zeros(x.shape[1]+1)
    for b in models:
        final_b+=learning_rate*b
    return final_b

def evaluate_xgboost_linear_regression(x,y,predictors,n_estimator,learning_rate):
    train_x, test_x ,train_y,test_y= train_test_split(x,y, test_size=0.2, random_state=42)
    models=xgboost_linear_regression_fit(train_x,train_y,n_estimator,learning_rate)
    print(models)
    final_b=xgboost_linear_regression_predict(models,train_x,learning_rate)
    print(final_b)
    y_train_pred=linear_regression_predict(train_x,final_b)
    y_test_pred = linear_regression_predict(test_x,final_b)
    
    print("Estimated Coefficients :")
    for i in range(len(final_b)):
        if i==0:
            print(f"Intercept : {final_b[i]:.4f}")
        else :
            print(f"Coefficient for {predictors[i-1]} : {final_b[i]:.4f}")
    # Plotting Between Test Actual vs Test Predicted
    plot_predictions_vs_actuals(test_y.values,y_test_pred,title="Linear Regression (Predicted vs Actual)")
    if len(x.shape) == 1:
        p = 1
    else:
        p = x.shape[1]
    # Metric Evaluation
    train_rss=compute_rss(train_y,y_train_pred)
    train_mse=compute_mse(train_rss,len(train_y))
    test_rss=compute_rss(test_y,y_test_pred)
    test_mse=compute_mse(test_rss,len(test_y))
    train_r2=r2_score(train_y,y_train_pred)
    test_r2=r2_score(test_y,y_test_pred)
    train_adj_r2=adjusted_r2_score(train_y,y_train_pred,p)
    test_adj_r2=adjusted_r2_score(test_y,y_test_pred,p)
    # Print All Metric Evaluation
    print(f"Train RSS : {train_rss :.4f}")
    print(f"Test RSS :{test_rss :.4f}")
    print(f"Train MSE :{train_mse :.4f}")
    print(f"Test MSE :{test_mse :.4f}")
    print(f"Train R^2 :{train_r2 :.2f}")
    print(f"Test R^2 :{test_r2 :.2f}")
    print(f"Train Adjusted R^2 :{train_adj_r2 :.2f}")
    print(f"Test Adjusted R^2 :{test_adj_r2 :.2f}")
