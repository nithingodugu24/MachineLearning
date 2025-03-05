import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression

# Create a dataset with 100 features, but only 5 are important
X, y = make_regression(
    n_samples=100, n_features=100, n_informative=5, noise=0.1, random_state=42
)

# Train Lasso Regression with different values of lambda
lambda_values = [0.01, 0.1, 1, 10]  # Small to large lambda values
plt.figure(figsize=(10, 6))

for i, alpha in enumerate(lambda_values):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    num_zero_coeffs = np.sum(lasso.coef_ == 0)

    plt.subplot(2, 2, i + 1)
    plt.bar(range(100), lasso.coef_)
    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient Value")
    plt.title(f"Lasso with λ={alpha} (Zero Coeffs: {num_zero_coeffs})")

plt.tight_layout()
plt.show()
