import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, f1_score
)
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import test
import torch
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import grad
from sklearn.metrics import mean_squared_error, r2_score
# --- Plotting Functions ---

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar=False, linewidths=1, linecolor='black')
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score_val = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', label=f"AUC = {auc_score_val:.4f}")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"AUC Score: {auc_score_val:.4f}")

def plot_coefficients_with_uncertainty(w_map, std_errors, predictors):
    labels = ["Bias"] + predictors
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(w_map)), w_map, yerr=std_errors, fmt='o', capsize=5)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.title("Posterior Mean and Uncertainty of Coefficients")
    plt.ylabel("Coefficient Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Helper Functions for Plotting Predictions with Uncertainty ---
def plot_with_uncertainty(y_true, y_mean, y_std, title="Predictions with Uncertainty"):
    x = np.arange(len(y_true))
    plt.figure(figsize=(10, 5))
    plt.plot(x, y_true, label="True", color="black")
    plt.plot(x, y_mean, label="Predicted Mean", color="blue")
    plt.fill_between(x, y_mean - 2 * y_std, y_mean + 2 * y_std, color="blue", alpha=0.2, label="95% CI")
    plt.title(title)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Target")
    plt.legend()
    plt.show()

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

# Helper function for Bayesian Linear Regression
def compute_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


def adjusted_r2_score(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
# --- Core Logistic Regression Functions ---

def sigmoid(x):
    return expit(x)

def neg_log_likelihood(beta, X, y, lambda_):
    z = X @ beta
    p = sigmoid(z)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    log_likelihood = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    regularization = (lambda_ / 2) * np.sum(beta[1:] ** 2)
    return log_likelihood + regularization

def compute_gradient(beta, X, y, lambda_):
    z = X @ beta
    p = sigmoid(z)
    grad = (X.T @ (p - y)) / len(y)
    grad[1:] += lambda_ * beta[1:]
    return grad

def logistic_regression(X, y, lambda_=1.0):
    X = np.c_[np.ones(X.shape[0]), X]
    beta_init = np.random.randn(X.shape[1]) * 0.01
    res = minimize(neg_log_likelihood, beta_init, args=(X, y, lambda_),
                   jac=compute_gradient, method='BFGS')
    return res.x, -res.fun, res.success

def logistic_regression_prediction(X, beta, threshold=0.5):
    X = np.c_[np.ones(X.shape[0]), X]
    p = sigmoid(X @ beta)
    y_pred = (p >= threshold).astype(int)
    return y_pred, p

# --- Evaluation Function ---

def evaluate_logistic_regression(X_raw, y, predictors, lambda_=1.0):
    assert X_raw.shape[0] == y.shape[0], "Mismatch in number of samples between X and y"

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    beta_std, log_likelihood, success = logistic_regression(train_X, train_y, lambda_)
    y_train_pred, p_train = logistic_regression_prediction(train_X, beta_std)
    y_test_pred, p_test = logistic_regression_prediction(test_X, beta_std)

    if not success:
        print(" Warning: Optimization did not converge.")

    # --- Rescale Coefficients to Original Scale ---
    sigma = scaler.scale_
    mu = scaler.mean_
    beta_orig = np.zeros_like(beta_std)
    beta_orig[1:] = beta_std[1:] / sigma
    beta_orig[0] = beta_std[0] - np.sum((beta_std[1:] * mu) / sigma)

    # Print final coefficients
    print("\nRescaled Coefficients (in original feature scale):")
    print(f"Bias (Intercept): {beta_orig[0]:.4f}")
    for i, name in enumerate(predictors):
        print(f"Coefficient for {name}: {beta_orig[i+1]:.4f}")

    print(f"\nFinal Log-likelihood (on training set): {log_likelihood:.4f}")

    # --- Evaluation ---
    train_acc = accuracy_score(train_y, y_train_pred)
    test_acc = accuracy_score(test_y, y_test_pred)
    f1 = f1_score(test_y, y_test_pred)

    print(f"\nTrain Accuracy: {train_acc:.2f}")
    print(f"Test Accuracy: {test_acc:.2f}")
    print(f"F1 Score (Test): {f1:.2f}")
    print("\nConfusion Matrix (Test):\n", confusion_matrix(test_y, y_test_pred))
    print("Classification Report (Test):\n", classification_report(test_y, y_test_pred))

    plot_confusion_matrix(test_y, y_test_pred, title="Logistic Regression - Confusion Matrix")
    plot_roc_curve(test_y, p_test, title="Logistic Regression - ROC Curve")


# Bayesian logistic Regression functions
# This implementation uses Laplace approximation for Bayesian logistic regression. 
def bayesian_logistic_regression_laplace(X, y, lambda_=1.0):
    N, D = X.shape
    X = np.c_[np.ones(N), X]  # Add bias
    D += 1

    def nll(w):
        z = X @ w
        p = sigmoid(z)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        log_likelihood = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        prior_penalty = (lambda_ / 2) * np.sum(w[1:] ** 2)
        return log_likelihood + prior_penalty

    def grad(w):
        z = X @ w
        p = sigmoid(z)
        g = X.T @ (p - y)
        g[1:] += lambda_ * w[1:]
        return g

    w_map = minimize(nll, np.zeros(D), jac=grad, method="BFGS").x

    z = X @ w_map
    p = sigmoid(z)
    S_diag = p * (1 - p)
    S = np.diag(S_diag)
    H = X.T @ S @ X + lambda_ * np.eye(D)
    cov = np.linalg.inv(H)
    return w_map, cov

# Bayesian logistic regression using Variational Inference (VI)
def bayesian_logistic_regression_vi(X, y, lambda_=1.0, n_samples=10, n_iter=100, lr=0.01):
    N, D = X.shape
    X = np.c_[np.ones(N), X]  # Add bias
    D += 1

    # Initialize variational parameters
    mu = np.zeros(D)
    rho = np.ones(D) * -3.0  # sigma = softplus(rho)

    def softplus(x):
        return np.log1p(np.exp(x))

    for step in range(n_iter):
        sigma = softplus(rho)
        epsilons = np.random.randn(n_samples, D)
        ws = mu + sigma * epsilons  # Reparameterization trick

        # Monte Carlo estimate of expected log-likelihood
        log_lik = 0
        for i in range(n_samples):
            w = ws[i]
            z = X @ w
            p = sigmoid(z)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            log_lik += np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        log_lik /= n_samples

        # Analytic KL divergence between q(w|mu,sigma) and p(w|0,I)
        sigma_sq = sigma**2
        kl = 0.5 * np.sum(mu**2 + sigma_sq - 1 - np.log(sigma_sq)) + (lambda_ - 1) * 0.5 * np.sum(mu[1:]**2)

        # ELBO
        elbo = log_lik - kl

        # Estimate gradients (finite difference approximation)
        mu_grad = np.zeros_like(mu)
        rho_grad = np.zeros_like(rho)
        eps = 1e-5

        for i in range(D):
            # mu grad
            mu_eps = mu.copy()
            mu_eps[i] += eps
            ws_eps = mu_eps + sigma * epsilons
            log_lik_eps = 0
            for w in ws_eps:
                z = X @ w
                p = sigmoid(z)
                p = np.clip(p, 1e-15, 1 - 1e-15)
                log_lik_eps += np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            log_lik_eps /= n_samples
            kl_eps = 0.5 * np.sum(mu_eps**2 + sigma_sq - 1 - np.log(sigma_sq))
            elbo_eps = log_lik_eps - kl_eps
            mu_grad[i] = (elbo_eps - elbo) / eps

            # rho grad
            rho_eps = rho.copy()
            rho_eps[i] += eps
            sigma_eps = softplus(rho_eps)
            ws_eps = mu + sigma_eps * epsilons
            log_lik_eps = 0
            for w in ws_eps:
                z = X @ w
                p = sigmoid(z)
                p = np.clip(p, 1e-15, 1 - 1e-15)
                log_lik_eps += np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            log_lik_eps /= n_samples
            sigma_eps_sq = sigma_eps ** 2
            kl_eps = 0.5 * np.sum(mu**2 + sigma_eps_sq - 1 - np.log(sigma_eps_sq))
            elbo_eps = log_lik_eps - kl_eps
            rho_grad[i] = (elbo_eps - elbo) / eps

        # Gradient ascent
        mu += lr * mu_grad
        rho += lr * rho_grad

    sigma_diag = softplus(rho)
    cov_diag = np.diag(sigma_diag**2)
    return mu, cov_diag

def bayesian_logistic_regression_vi_torch(X_np, y_np, lambda_=1.0, n_samples=50, n_iter=1000, lr=0.01):
    X_np = np.c_[np.ones(X_np.shape[0]), X_np]  # Add bias
    N, D = X_np.shape

    # Convert to torch tensors
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    # Variational parameters (mean and rho for std)
    mu = torch.zeros(D, requires_grad=True)
    rho = torch.full((D,), -3.0, requires_grad=True)

    optimizer = torch.optim.Adam([mu, rho], lr=lr)

    def softplus(x):
        return torch.nn.functional.softplus(x)

    for step in range(n_iter):
        optimizer.zero_grad()

        sigma = softplus(rho)  # std = softplus(rho)
        q_dist = Normal(mu, sigma)

        # Reparameterization trick
        epsilons = torch.randn((n_samples, D))
        ws = mu + sigma * epsilons

        # Monte Carlo estimate of expected log-likelihood
        log_lik = 0.0
        for w in ws:
            z = X @ w
            p = torch.sigmoid(z)
            p = torch.clamp(p, 1e-15, 1 - 1e-15)
            log_lik += torch.sum(y * torch.log(p) + (1 - y) * torch.log(1 - p))
        log_lik /= n_samples

        # KL divergence from q(w) to p(w) ~ N(0, I)
        sigma_sq = sigma ** 2
        kl = 0.5 * torch.sum(mu**2 + sigma_sq - 1 - torch.log(sigma_sq))

        # ELBO (maximize)
        elbo = log_lik - lambda_ * kl
        loss = -elbo  # we minimize the negative ELBO

        loss.backward()
        optimizer.step()

    # Final posterior parameters
    mu_final = mu.detach().numpy()
    sigma_final = softplus(rho).detach().numpy()
    cov_diag = sigma_final ** 2
    return mu_final, np.diag(cov_diag)

# Bayesian logistic regression using MCMC with Hamiltonian Monte Carlo (HMC)
def bayesian_logistic_regression_mcmc_hmc(X_np, y_np, lambda_=1.0, n_samples=1000, step_size=0.01, n_leapfrog=10):
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)
    N, D = X.shape
    X = torch.cat([torch.ones(N, 1), X], dim=1)  # Add bias term
    D += 1

    def log_posterior(w):
        z = X @ w
        likelihood = torch.sum(y * torch.log(torch.sigmoid(z) + 1e-8) + (1 - y) * torch.log(1 - torch.sigmoid(z) + 1e-8))
        prior = -0.5 * lambda_ * torch.sum(w[1:] ** 2)
        return likelihood + prior

    def hmc_sample(current_w):
        w = current_w.clone().detach().requires_grad_(True)
        momentum = torch.randn_like(w)
        current_U = -log_posterior(w)
        current_K = 0.5 * torch.sum(momentum ** 2)

        # Leapfrog integration
        new_w = w.clone()
        new_momentum = momentum.clone()

        # Half step
        grad_U = grad(-log_posterior(new_w), new_w)[0]
        new_momentum -= 0.5 * step_size * grad_U

        # Full steps
        for _ in range(n_leapfrog):
            new_w = new_w + step_size * new_momentum
            new_w = new_w.detach().requires_grad_(True)
            grad_U = grad(-log_posterior(new_w), new_w)[0]
            if _ != n_leapfrog - 1:
                new_momentum -= step_size * grad_U

        # Final half step
        new_momentum -= 0.5 * step_size * grad_U

        new_w = new_w.detach()
        new_momentum = -new_momentum  # Negate for symmetry

        new_U = -log_posterior(new_w)
        new_K = 0.5 * torch.sum(new_momentum ** 2)

        accept_prob = torch.exp(current_U - new_U + current_K - new_K)
        if torch.rand(1) < accept_prob:
            return new_w.detach()
        else:
            return current_w.detach()

    samples = []
    w_init = torch.zeros((D, 1))
    current_w = w_init

    for i in range(n_samples + 100):  # Include burn-in
        current_w = hmc_sample(current_w)
        if i >= 100:
            samples.append(current_w.view(-1).numpy())

    samples = np.array(samples)
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples.T)
    return mean, cov

# Bayesian logistic regression using MCMC with Metropolis-Hastings

def bayesian_logistic_regression_mcmc_metropolis(X, y, lambda_=1.0, n_samples=1000, proposal_std=0.05, burn_in=100):
    N, D = X.shape
    X = np.c_[np.ones(N), X]  # Add bias
    D += 1

    def log_posterior(w):
        z = X @ w
        p = sigmoid(z)
        log_likelihood = np.sum(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))
        log_prior = -0.5 * lambda_ * np.sum(w[1:] ** 2)
        return log_likelihood + log_prior

    samples = []
    w = np.zeros(D)

    for i in range(n_samples + burn_in):
        proposal = w + np.random.normal(0, proposal_std, size=D)
        log_alpha = log_posterior(proposal) - log_posterior(w)
        if np.log(np.random.rand()) < log_alpha:
            w = proposal
        if i >= burn_in:
            samples.append(w)

    samples = np.array(samples)
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples.T)
    return mean, cov

# Bayesian logistic regression prediction function for laplce and VI approximation
def bayesian_logistic_prediction(X, w_map, cov, n_samples=1000):
    N = X.shape[0]
    X = np.c_[np.ones(N), X]
    samples = np.random.multivariate_normal(w_map, cov, size=n_samples)
    prob_samples = sigmoid(X @ samples.T)
    y_pred_mean = np.mean(prob_samples, axis=1)
    y_pred = (y_pred_mean >= 0.5).astype(int)
    return y_pred, y_pred_mean

# --- Main Evaluation Function for Bayesian Logistic Regression ---
def evaluate_bayesian_logistic_regression(X_raw, y, predictors, method="laplace", lambda_=1.0):
    assert X_raw.shape[0] == y.shape[0], "Mismatch in number of samples and labels."

    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit posterior
    if method == "laplace":
        w_map, cov = bayesian_logistic_regression_laplace(train_X, train_y, lambda_)
        y_test_pred, y_test_probs = bayesian_logistic_prediction(test_X, w_map, cov)
        y_train_pred, y_train_probs = bayesian_logistic_prediction(train_X, w_map, cov)
    elif method == "vi":
        w_map, cov = bayesian_logistic_regression_vi(train_X, train_y, lambda_=lambda_)
        y_test_pred, y_test_probs = bayesian_logistic_prediction(test_X, w_map, cov)
        y_train_pred, y_train_probs = bayesian_logistic_prediction(train_X, w_map, cov)
    elif method == "vi_torch":
        train_X = np.asarray(train_X)
        train_y = np.asarray(train_y)
        w_map, cov = bayesian_logistic_regression_vi_torch(train_X, train_y, lambda_=lambda_)
        y_test_pred, y_test_probs = bayesian_logistic_prediction(test_X, w_map, cov)
        y_train_pred, y_train_probs = bayesian_logistic_prediction(train_X, w_map, cov)
    elif method == "mcmc_hmc":
        train_X = np.asarray(train_X)
        train_y = np.asarray(train_y)
        w_map, cov = bayesian_logistic_regression_mcmc_hmc(train_X, train_y, lambda_=lambda_)
        y_test_pred, y_test_probs = bayesian_logistic_prediction(test_X, w_map, cov)
        y_train_pred, y_train_probs = bayesian_logistic_prediction(train_X, w_map, cov)
    elif method == "mcmc_metropolis":
        train_X = np.asarray(train_X)
        train_y = np.asarray(train_y)
        w_map, cov = bayesian_logistic_regression_mcmc_metropolis(train_X, train_y, lambda_=lambda_)
        y_test_pred, y_test_probs = bayesian_logistic_prediction(test_X, w_map, cov)
        y_train_pred, y_train_probs = bayesian_logistic_prediction(train_X, w_map, cov)
    else:
        raise NotImplementedError(f"Method '{method}' not implemented yet.")

    # --- Uncertainty Estimation ---
    cov_diag= np.diag(cov)
    std_errors = np.sqrt(cov_diag)
    # --- Rescale coefficients to original feature scale ---
    sigma = scaler.scale_
    mu = scaler.mean_
    w_rescaled = np.zeros_like(w_map)
    w_rescaled[1:] = w_map[1:] / sigma
    w_rescaled[0] = w_map[0] - np.sum((w_map[1:] * mu) / sigma)

    std_rescaled = np.zeros_like(std_errors)
    std_rescaled[1:] = std_errors[1:] / sigma
    std_rescaled[0] = std_errors[0]  # bias unaffected by scaling

    # --- Coefficient Output ---
    print("\nPosterior Mean Coefficients (Rescaled with Uncertainty):")
    print(f"{'Bias (Intercept)':>20}: {w_rescaled[0]: .4f} ± {std_rescaled[0]: .4f}")
    for i, name in enumerate(predictors):
        print(f"{name:>20}: {w_rescaled[i + 1]: .4f} ± {std_rescaled[i + 1]: .4f}")

    # --- Classification Evaluation ---
    test_acc = accuracy_score(test_y, y_test_pred)
    train_acc = accuracy_score(train_y, y_train_pred)
    f1 = f1_score(test_y, y_test_pred)
    print(f"\nTrain Accuracy: {train_acc:.2f}")
    print(f"\nTest Accuracy: {test_acc:.2f}")

    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:\n", confusion_matrix(test_y, y_test_pred))
    print("Classification Report:\n", classification_report(test_y, y_test_pred))

    # --- Visuals ---
    plot_confusion_matrix(test_y, y_test_pred, title=f"Bayesian Logistic Regression ({method}) - Confusion Matrix")
    plot_roc_curve(test_y, y_test_probs, title=f"Bayesian Logistic Regression ({method}) - ROC Curve")
    plot_coefficients_with_uncertainty(w_rescaled, std_rescaled, predictors)

# Now i am going to write function for Bayesian linear regression
def bayesian_linear_regression(X, y, sigma2=1.0, tau2=1.0):
    N, D = X.shape
    X = np.c_[np.ones(N), X]  # Add bias term
    D += 1

    I = np.eye(D)
    I[0, 0] = 0  # Do not regularize bias term

    XT_X = X.T @ X
    XT_y = X.T @ y

    cov = np.linalg.inv((1 / sigma2) * XT_X + (1 / tau2) * I)
    mean = (1 / sigma2) * cov @ XT_y

    return mean, cov


# Bayesian linear regression prediction function
def bayesian_linear_prediction(X, w_post, cov, sigma2=1.0):
    N = X.shape[0]
    X = np.c_[np.ones(N), X]  # Add bias term
    mean = X @ w_post

    # Compute predictive variance (per point)
    variance = np.einsum('ij,jk,ik->i', X, cov, X) + sigma2
    std = np.sqrt(variance)
    return mean, std

# --- Main Evaluation Function for Bayesian Linear Regression ---
def evaluate_bayesian_linear_regression(X_raw, y, predictors, lambda_=1.0):
    assert X_raw.shape[0] == y.shape[0], "Mismatch in number of samples and labels."

    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit posterior
    w_map, cov = bayesian_linear_regression(train_X, train_y, lambda_)

    # Estimate residual variance from training data
    X_train_bias = np.c_[np.ones(train_X.shape[0]), train_X]
    sigma2 = np.var(train_y - X_train_bias @ w_map)


    # Predict with uncertainty
    y_test_pred, y_test_std = bayesian_linear_prediction(test_X, w_map, cov, sigma2=sigma2)
    y_train_pred, y_train_std = bayesian_linear_prediction(train_X, w_map, cov, sigma2=sigma2)

    # --- Uncertainty in coefficients ---
    cov_diag = np.diag(cov)
    std_errors = np.sqrt(cov_diag)

    # --- Rescale coefficients to original feature scale ---
    sigma = scaler.scale_
    mu = scaler.mean_
    w_rescaled = np.zeros_like(w_map)
    w_rescaled[1:] = w_map[1:] / sigma
    w_rescaled[0] = w_map[0] - np.sum((w_map[1:] * mu) / sigma)

    std_rescaled = np.zeros_like(std_errors)
    std_rescaled[1:] = std_errors[1:] / sigma
    std_rescaled[0] = std_errors[0]
    print("="*50)
    print("\nResults of Bayesian Linear Regression\n")
    # --- Coefficient Output ---
    print("\nPosterior Mean Coefficients (Rescaled with Uncertainty):")
    print(f"{'Bias (Intercept)':>20}: {w_rescaled[0]: .4f} ± {std_rescaled[0]: .4f}")
    for i, name in enumerate(predictors):
        print(f"{name:>20}: {w_rescaled[i + 1]: .4f} ± {std_rescaled[i + 1]: .4f}")

    # --- Evaluation Metrics ---
    rss_train = compute_rss(train_y, y_train_pred)
    rss_test = compute_rss(test_y, y_test_pred)
    train_mse = mean_squared_error(train_y, y_train_pred)
    test_mse = mean_squared_error(test_y, y_test_pred)
    train_r2 = r2_score(train_y, y_train_pred)
    test_r2 = r2_score(test_y, y_test_pred)

    print("\nEvaluation Metrics:")
    print(f"RSS (Train): {rss_train:.4f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"RSS (Test): {rss_test:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Train R^2: {train_r2:.4f}")
    print(f"Test R^2: {test_r2:.4f}")

    # --- Plots ---
    plot_predictions_vs_actuals(train_y,y_train_pred,title="Bayesian Linear Regression - Predictions vs Actuals on Train Data")
    plot_predictions_vs_actuals(test_y, y_test_pred, title="Bayesian Linear Regression - Predictions vs Actuals on Test Data")
    plot_coefficients_with_uncertainty(w_rescaled, std_rescaled, predictors)
    plot_with_uncertainty(test_y, y_test_pred, y_test_std, title="Bayesian Linear Regression - Predictions with Uncertainty")
