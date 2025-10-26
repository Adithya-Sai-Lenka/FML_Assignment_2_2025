import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots/question2', exist_ok=True)


dataset_train = np.genfromtxt('A2Q2train.csv', delimiter=',')
X = dataset_train[:, :-1].T
y = dataset_train[:, -1]


## i. Obtain the Least Squares solution W_ML using the analytical solution
w_ML = np.linalg.inv(X @ X.T) @ X @ y



np.random.seed(42)
w_init = np.random.randn(X.shape[0])



## ii. Code Gradient Descent (GD) to obtain w_GD
w = w_init.copy()
learning_rate = 0.02
distances_gd = []
distances_gd.append(np.linalg.norm(w - w_ML))
for i in range(5000):
    gradient = 2 * X @ (X.T @ w - y) / X.shape[1]
    w -= learning_rate * gradient
    distances_gd.append(np.linalg.norm(w - w_ML))

plt.figure(figsize=(8, 6))
plt.plot(distances_gd)
plt.xlabel('Iterations (t)')
plt.ylabel('Distance from w_ML (||w(t) - w_ML||)')
plt.title('Gradient Descent Convergence')
plt.grid(True)
plt.savefig('plots/question2/gd_convergence.png')
plt.close()

## iii. Code Stochastic Gradient Descent (SGD) to obtain w_SGD
w = w_init.copy()
learning_rate = 0.02
distances_sgd = []
distances_sgd.append(np.linalg.norm(w - w_ML))

total_samples = X.shape[1]
batch_size = 100

for i in range(5000):
    column_indices = np.random.choice(total_samples, batch_size, replace=False)
    X_batch = X[:, column_indices]
    y_batch = y[column_indices]
    gradient = 2 * X_batch @ (X_batch.T @ w - y_batch) / X_batch.shape[1]
    w -= learning_rate * gradient
    distances_sgd.append(np.linalg.norm(w - w_ML))

plt.figure(figsize=(8, 6))
plt.plot(distances_sgd)
plt.xlabel('Iterations (t)')
plt.ylabel('Distance from w_ML (||w(t) - w_ML||)')
plt.title('Stochastic Gradient Descent Convergence')
plt.grid(True)
plt.savefig('plots/question2/sgd_convergence.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(distances_gd[:50], label='Gradient Descent')
plt.plot(distances_sgd[:50], label='Stochastic Gradient Descent')
plt.xlabel('Iterations (t)')
plt.ylabel('Distance from w_ML (||w(t) - w_ML||)')
plt.title('Convergence Comparison (start)')
plt.grid(True)
plt.legend()
plt.savefig('plots/question2/convergence_comparison_start.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(np.arange(4950,5000), distances_gd[-50:], label='Gradient Descent')
plt.plot(np.arange(4950,5000), distances_sgd[-50:], label='Stochastic Gradient Descent')
plt.xlabel('Iterations (t)')
plt.ylabel('Distance from w_ML (||w(t) - w_ML||)')
plt.title('Convergence Comparison (end)')
plt.grid(True)
plt.legend()
plt.savefig('plots/question2/convergence_comparison_end.png')
plt.close()

## iv. Gradient Descent for Ridge Regression and cross validation to find best lambda

val_cols = np.random.choice(X.shape[1], 200, replace=False)

X_train = np.delete(X, val_cols, axis=1)
X_val = X[:, val_cols]
y_train = np.delete(y, val_cols, axis=0)
y_val = y[val_cols]


lambda_coeffs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
val_errors = {}
w_from_ridge = {}
for lambda_coeff in lambda_coeffs:
    w = w_init.copy()
    learning_rate = 0.02
    for i in range(5000):
        gradient = 2 * X_train @ (X_train.T @ w - y_train) / X_train.shape[1] + lambda_coeff*w
        w -= learning_rate * gradient
    val_error = np.mean((y_val - w.T @ X_val) ** 2)
    val_errors[str(lambda_coeff)] = val_error
    w_from_ridge[str(lambda_coeff)] = w

plt.figure(figsize=(8, 6))
plt.plot(list(val_errors.keys()), list(val_errors.values()), marker='o')
plt.xlabel('Lambda')
plt.ylabel('Validation MSE')
plt.title('Ridge Regression: Validation MSE vs Lambda')
plt.grid(True)
plt.savefig('plots/question2/ridge_validation_mse.png')
plt.close()

print("Best lambda:", min(val_errors, key=val_errors.get))

## Testing

dataset_test = np.genfromtxt('A2Q2test.csv', delimiter=',')
X_test = dataset_test[:, :-1].T
y_test = dataset_test[:, -1]

test_error_ML = np.mean((y_test - w_ML.T @ X_test) ** 2)
test_error_ridge = np.mean((y_test - w_from_ridge[min(val_errors, key=val_errors.get)].T @ X_test) ** 2)

print("Test MSE for Least Squares (w_ML):", test_error_ML)
print("Test MSE for Ridge Regression (best lambda):", test_error_ridge)