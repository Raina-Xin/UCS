from collections import defaultdict, Counter
import numpy as np
from scipy.stats import nbinom
from scipy.optimize import minimize
import scipy
from scipy.stats import poisson

################### Step 4: Count Knowledge Frequencies ################### 
def count_knowledge_frequencies(clustered_names):
    """Counts the number of times each knowledge instance appears."""
    return Counter(clustered_names.values())


def smooth_counts_powerlaw(counts, k=None):
    from scipy.optimize import curve_fit
    if k is None:
        k = len(counts)
    r_nonzero = np.arange(1, len(counts) + 1)

    # Power-law function
    def power_law(r, C, alpha, beta):
        return C * (beta+r) ** (-alpha)

    # Fit the power-law model
    params, _ = curve_fit(power_law, r_nonzero[:k], counts[:k])
    C_fit, alpha_fit, beta_fit = params

    # Get the fitted data
    fitted_data = power_law(r_nonzero, C_fit, alpha_fit, beta_fit)

    return fitted_data

def smooth_counts_poisson(counts, k=None):
    counts = np.asarray(counts)
    if k is None:
        k = len(counts)
    s_vals = np.arange(1, len(counts) + 1)
    N = np.sum(counts[:k])  # total number of distinct types
    T = np.sum(s_vals[:k] * counts[:k])  # total number of tokens
    
    # MLE for λ is just average frequency per type
    lambda_mle = T / N

    # Generate fitted spectrum using Poisson PMF (no need to re-optimize if using MLE)
    fitted_counts = T * poisson.pmf(s_vals, mu=lambda_mle)
    # print()
    return fitted_counts

def smooth_counts_binomial(observed_counts):
    # Rank positions
    ranks = np.arange(1, len(observed_counts) + 1)

    # Negative Binomial PMF model
    def negative_binomial_pmf(r, r_param, p_param, scale):
        return scale * nbinom.pmf(r - 1, r_param, p_param)

    # Loss function to minimize (sum of squared differences)
    def loss(params):
        r_param, p_param, scale = params
        fitted = negative_binomial_pmf(ranks, r_param, p_param, scale)
        return np.sum((observed_counts - fitted) ** 2)

    # Initial guesses for parameters
    initial_params = [10, 0.5, 500]

    # Bounds to ensure parameters are valid
    bounds = [(1e-5, None), (1e-5, 1 - 1e-5), (1e-5, None)]

    # Optimize parameters
    result = minimize(loss, initial_params, bounds=bounds)
    r_fit, p_fit, scale_fit = result.x

    # Fitted values
    fitted_counts = negative_binomial_pmf(ranks, r_fit, p_fit, scale_fit)
    return fitted_counts

def smoothed_good_toulmin_sgt(counts, t_list, bin_size=20, smooth_count=False, adaptive=False, mute=False, offset=1):
    """Estimates unseen knowledge using the Smoothed Good-Toulmin estimator with the given SGT formula."""
    assert 1<= offset <= 2
    if type(t_list) is int:
        t_list = [t_list]
    freq_counts = defaultdict(int, counts)
    n_seen = np.sum([appear * number for (appear, number) in counts.items() ])
    n_knowledge = np.sum([number for (appear, number) in counts.items() ])

    if not mute:
        print(f"    With N={n_seen}, # seen = {n_knowledge}\n")

    if smooth_count:
        freq_spectrum = np.array([freq_counts[k] for k in range(1,1+len(freq_counts))])
        fitted_counts = smooth_counts_powerlaw(freq_spectrum)
        freq_counts = {i:int(fitted_counts[i-1]) for i in np.arange(1, len(fitted_counts) + 1)}

    def estimate_for_bin_size(bin_size):
        unseen_estimates = []
        unseen_stds = []
        for t in t_list:
            unseen_estimate = 0
            unseen_variance = 0
            for s in range(1, bin_size + 1):
                freq_s = freq_counts[s]
                probability = 1 - scipy.stats.binom.cdf(s - 1, bin_size, offset / (t + offset))
                delta = - (-t)** s * probability
                unseen_estimate += delta * freq_s
                unseen_variance += delta**2 * freq_s
            unseen_estimate = int(max(0, unseen_estimate))  # Ensure non-negative
            unseen_std = int(np.sqrt(max(0, unseen_variance)))  # Ensure non-negative
            unseen_estimates.append(unseen_estimate)
            unseen_stds.append(unseen_std)

            if not mute:
                print(f"    With N={n_seen}, t={t}, # unseen = {unseen_estimate} with std = {unseen_std}")
        return unseen_estimates, unseen_stds
    
    return estimate_for_bin_size(bin_size)
