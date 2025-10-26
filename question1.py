import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots/question1', exist_ok=True)

dataset = np.genfromtxt('A2Q1.csv', delimiter=',')



### EM for Bernoulli Mixture Model

def bernoulli_mixture_em(X, K=4, max_iter=50, random_state=42):
    np.random.seed(random_state)
    N, D = X.shape

    # Initialize parameters
    pi = np.ones(K) / K
    mu = np.random.rand(K, D) * 0.5 + 0.25  # Avoid extreme values

    log_likelihoods = []

    for iteration in range(max_iter):
        # E-step
        resp = np.zeros((N, K))

        ll = 0
        for i in range(N):
            for k in range(K):
                prob = np.prod(mu[k]**X[i] * (1 - mu[k])**(1 - X[i]))
                resp[i, k] = pi[k] * prob

            ll += np.log(resp[i].sum())
        resp /= resp.sum(axis=1, keepdims=True)

        log_likelihoods.append(ll)

        # M-step
        N_k = resp.sum(axis=0)
        pi = N_k / N
        mu = (resp.T @ X) / N_k[:, np.newaxis]

    # Final LL computation
    resp = np.zeros((N, K))
    ll = 0
    for i in range(N):
        for k in range(K):
            prob = np.prod(mu[k]**X[i] * (1 - mu[k])**(1 - X[i]))
            resp[i, k] = pi[k] * prob
        ll+= np.log(resp[i].sum())
    log_likelihoods.append(ll)
    
    return pi, mu, log_likelihoods
    

N_init = 100
np.random.seed(42)
seeds = np.random.randint(0, 100, N_init)

bmm_lls = []
for i in range(N_init):
    pi, mu, log_likelihoods = bernoulli_mixture_em(dataset, K=4, max_iter=25, random_state=seeds[i])
    bmm_lls.append(log_likelihoods)

bmm_lls = np.array(bmm_lls)
bmm_lls = bmm_lls.mean(axis=0)

plt.figure(figsize=(8, 6))
plt.plot(bmm_lls)
plt.xlabel('Iterations')
plt.ylabel('Average Log-Likelihood')
plt.title('EM for Bernoulli Mixture Model Convergence')
plt.grid(True)
plt.savefig('plots/question1/bernoulli_mixture_em_convergence.png')
plt.close()


### EM for Gaussian Mixture Model (with Diagonal Covariance Matrices)

def gaussian_mixture_em(X, K=4, max_iter=50, random_state=42):
    np.random.seed(random_state)
    N, D = X.shape

    # Initialize parameters
    pi = np.ones(K) / K
    # Choose random data points as initial means
    mu = X[np.random.choice(N, K, replace=False)]
    # Initialize covariances to identity matrices
    sigma = np.array([np.eye(D) for _ in range(K)]) ## Assume given the mixture, features are independent

    log_likelihoods = []

    # Epsilon for numerical stability
    EPS = 1e-9 # To prevent singular covariance matrices during inversion
    reg_covariance = 1e-6

    for iteration in range(max_iter):
        # E-step
        resp = np.zeros((N, K))

        inv_sigma = np.zeros_like(sigma)
        dets = np.zeros(K)
        coeffs = np.zeros(K)
        
        for k in range(K):
            # Rely on regularization in M-step during previous iteration to ensure sigma is invertible
            inv_sigma[k] = np.linalg.inv(sigma[k])
            dets[k] = np.linalg.det(sigma[k])
            coeffs[k] = 1 / np.sqrt((2 * np.pi) ** D * dets[k])

        ll = 0
        for i in range(N):
            for k in range(K):
                diff = X[i] - mu[k]
                exponent = -0.5 * diff.T @ inv_sigma[k] @ diff
                prob = coeffs[k] * np.exp(exponent)
                resp[i, k] = pi[k] * prob

            row_sum = resp[i].sum()
            # Handle numerical underflow where prob is 0 for all k
            if row_sum > EPS:
                ll += np.log(row_sum)

        resp_sum = resp.sum(axis=1, keepdims=True)    
        resp /= (resp_sum + EPS)  # Avoid division by zero

        log_likelihoods.append(ll)

        # M-step
        N_k = resp.sum(axis=0)
        pi = N_k / N
        mu = (resp.T @ X) / (N_k[:, np.newaxis]+ EPS)

        for k in range(K):
            diff = X - mu[k]
            sigma[k] = (resp[:, k][:, np.newaxis] * diff).T @ diff / (N_k[k] + EPS)

            sigma[k] += reg_covariance * np.eye(D)  # Regularization to ensure positive definiteness

    # Final LL computation
    inv_sigma = np.zeros_like(sigma)
    dets = np.zeros(K)
    coeffs = np.zeros(K)
    for k in range(K):
        inv_sigma[k] = np.linalg.inv(sigma[k])
        dets[k] = np.linalg.det(sigma[k])
        coeffs[k] = 1 / np.sqrt((2 * np.pi) ** D * dets[k])

    final_ll = 0

    for i in range(N):
        prob_i = 0
        for k in range(K):
            diff = X[i] - mu[k]
            exponent = -0.5 * diff.T @ inv_sigma[k] @ diff
            prob = coeffs[k] * np.exp(exponent)
            prob_i += pi[k] * prob
        
        if prob_i > EPS:
            final_ll += np.log(prob_i)

    log_likelihoods.append(final_ll)

    return pi, mu, sigma, log_likelihoods

N_init = 100
np.random.seed(42)
seeds = np.random.randint(0, 100, N_init)

gmm_lls = []
for i in range(N_init):
    pi, mu, sigma, log_likelihoods = gaussian_mixture_em(dataset, K=4, max_iter=25, random_state=seeds[i])
    gmm_lls.append(log_likelihoods)

gmm_lls = np.array(gmm_lls)
gmm_lls = gmm_lls.mean(axis=0)

plt.figure(figsize=(8, 6))
plt.plot(gmm_lls)
plt.xlabel('Iterations')
plt.ylabel('Average Log-Likelihood')
plt.title('EM for Gaussian Mixture Model Convergence')
plt.grid(True)
plt.savefig('plots/question1/gaussian_mixture_em_convergence.png')
plt.close()


def gaussian_mixture_em_robust(X, K=4, max_iter=50, random_state=42, reg_covar=1e-14):
    """
    Numerically stable implementation of EM for GMM using log-probabilities.
    """
    np.random.seed(random_state)
    N, D = X.shape

    # Initialize parameters
    pi = np.ones(K) / K
    mu = X[np.random.choice(N, K, replace=False)]
    sigma = np.array([np.eye(D) for _ in range(K)])

    log_likelihoods = []
    EPS = 1e-9 # For numerical stability

    for iteration in range(max_iter):
        
        # --- E-step (in Log-Space) ---
        
        # log_resp shape (N, K)
        # We will compute log(pi_k * N(x_i | mu_k, sigma_k))
        # = log(pi_k) + log(N(x_i | mu_k, sigma_k))
        log_resp = np.zeros((N, K))

        for k in range(K):
            # Pre-calculate inverse and log-determinant
            try:
                inv_sigma_k = np.linalg.inv(sigma[k])
                # slogdet returns (sign, log_det). 
                # Since sigma is positive definite, sign is 1.
                sign, log_det_sigma_k = np.linalg.slogdet(sigma[k])
                
                # This check handles the (rare) case where slogdet still fails
                if sign < 1 or not np.isfinite(log_det_sigma_k):
                    raise np.linalg.LinAlgError
            
            except np.linalg.LinAlgError:
                # If matrix is singular even with regularization,
                # this component is dead. Assign it -inf log-prob.
                log_resp[:, k] = -np.inf
                continue

            # This is log(coeff) from your original code
            # log( 1 / sqrt((2*pi)^D * det(sigma)) )
            # = -0.5 * (D * log(2*pi) + log_det(sigma))
            log_coeff = -0.5 * (D * np.log(2 * np.pi) + log_det_sigma_k)

            # This is the log(exponent) part for all N points at once
            # -0.5 * (X - mu_k).T @ inv(sigma_k) @ (X - mu_k)
            diff = X - mu[k]
            # (diff @ inv_sigma_k) has shape (N, D)
            # np.sum(..., axis=1) performs the dot product for each row
            log_exponent = -0.5 * np.sum((diff @ inv_sigma_k) * diff, axis=1)

            # log(pi_k) + log(N(x_i | ...))
            log_resp[:, k] = np.log(pi[k] + EPS) + log_coeff + log_exponent

        # Now, normalize the log_resp matrix to get responsibilities
        # This is the "log-sum-exp" trick for numerical stability
        
        # 1. Find the maximum log-prob for each data point
        log_resp_max = np.max(log_resp, axis=1, keepdims=True)
        
        # 2. Subtract max to prevent overflow when exponentiating
        # (This handles rows that might be all -inf)
        log_resp_shifted = log_resp - log_resp_max
        with np.errstate(over='ignore', under='ignore'):
             exp_log_resp_shifted = np.exp(log_resp_shifted)

        # 3. Sum the probabilities (which are now between 0 and 1)
        resp_sum = np.sum(exp_log_resp_shifted, axis=1, keepdims=True)
        
        # 4. Normalize to get responsibilities
        resp = exp_log_resp_shifted / (resp_sum + EPS)
        
        # Calculate log-likelihood for this iteration
        # LL = sum_i( log(sum_k( pi_k * N_k )) )
        # Using the log-sum-exp trick again:
        # LL = sum_i( log_resp_max_i + log(sum_k(exp(log_resp_shifted_ik))) )
        # LL = sum_i( log_resp_max_i + log(resp_sum_i) )
        ll = np.sum(log_resp_max.ravel() + np.log(resp_sum.ravel() + EPS))
        log_likelihoods.append(ll)


        # --- M-step (Unchanged) ---
        N_k = resp.sum(axis=0)
        
        # Add EPS to N_k to prevent division by zero if a component dies
        N_k_stable = N_k + EPS
        
        pi = N_k / N
        mu = (resp.T @ X) / N_k_stable[:, np.newaxis]
        
        for k in range(K):
            diff = X - mu[k]
            sigma[k] = (resp[:, k][:, np.newaxis] * diff).T @ diff / N_k_stable[k]
            
            # Add regularization
            sigma[k] += np.eye(D) * reg_covar

    # Final LL is just the last one we computed
    if log_likelihoods:
        log_likelihoods.append(log_likelihoods[-1])
    else:
        # Handle case where max_iter = 0
        log_likelihoods.append(np.nan)


    return pi, mu, sigma, log_likelihoods

N_init = 100
np.random.seed(42)
seeds = np.random.randint(0, 100, N_init)

rob_gmm_lls = []
for i in range(N_init):
    pi, mu, sigma, log_likelihoods = gaussian_mixture_em_robust(dataset, K=4, max_iter=25, random_state=seeds[i])
    rob_gmm_lls.append(log_likelihoods)

rob_gmm_lls = np.array(rob_gmm_lls)
rob_gmm_lls = rob_gmm_lls.mean(axis=0)

plt.figure(figsize=(8, 6))
plt.plot(rob_gmm_lls)
plt.xlabel('Iterations')
plt.ylabel('Average Log-Likelihood')
plt.title('Robust EM for Gaussian Mixture Model Convergence')
plt.grid(True)
plt.savefig('plots/question1/robust_gaussian_mixture_em_convergence.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(bmm_lls, label='Bernoulli Mixture Model')
plt.plot(gmm_lls, label='Gaussian Mixture Model')
plt.title('Convergence Comparison')
plt.xlabel('Iterations')
plt.ylabel('Average Log-Likelihood')
plt.legend()
plt.grid(True)
plt.savefig('plots/question1/convergence_comparison.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(bmm_lls, label='Bernoulli Mixture Model')
plt.plot(rob_gmm_lls, label='Robust Gaussian Mixture Model')
plt.title('Convergence Comparison with Robust GMM')
plt.xlabel('Iterations')
plt.ylabel('Average Log-Likelihood')
plt.legend()
plt.grid(True)
plt.savefig('plots/question1/convergence_comparison_robust.png')
plt.close() 

def kmeans(X, K=4, max_iter=50, random_state=42):
    np.random.seed(random_state)
    N, D = X.shape

    # Initialize centroids by randomly selecting K data points
    centroids = X[np.random.choice(N, K, replace=False)]
    
    objective_values = [] # Sum of squared distances from cluster centroid

    for iteration in range(max_iter):
        distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)  # Shape (N, K)
        cluster_assignments = np.argmin(distances, axis=1)  # Shape (N,)

        obj = distances[np.arange(N), cluster_assignments].sum()
        objective_values.append(obj)

        for k in range(K):
            points_in_cluster = X[cluster_assignments == k]
            if len(points_in_cluster) > 0:
                centroids[k] = points_in_cluster.mean(axis=0)
            else:
                # If a cluster gets no points, reinitialize its centroid randomly
                centroids[k] = X[np.random.choice(N)]
    
    return centroids, cluster_assignments, objective_values    


N_init = 10
np.random.seed(42)
seeds = np.random.randint(0, 100, N_init)

objective_values_list = []
for i in range(N_init):
    centroids, assignments, objective_values = kmeans(dataset, K=4, max_iter=25, random_state=seeds[i])
    objective_values_list.append(objective_values)

objective_values_list = np.array(objective_values_list)
avg_obj_values = objective_values_list.mean(axis=0)

plt.figure(figsize=(8, 6))
plt.plot(avg_obj_values)
plt.xlabel('Iterations')
plt.ylabel('Average Objective Value')
plt.title('K-Means Convergence')
plt.grid(True)
plt.savefig('plots/question1/kmeans_convergence.png')
plt.close()