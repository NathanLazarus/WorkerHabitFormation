import numpy as np
from casadi import *


def fitted_differentiable_ls_function_numpy(wage, logit_center_vals, ols_coefs, poly_degree):
    poly_entries = wage[:,None] ** np.arange(poly_degree + 1)
    logit1 = 1 / (1 + np.exp(-1 * (wage[:,None] - logit_center_vals)))
    logit2 = 1 / (1 + np.exp(-2 * (wage[:,None] - logit_center_vals)))
    logit3 = 1 / (1 + np.exp(-3 * (wage[:,None] - logit_center_vals)))
    logit4 = 1 / (1 + np.exp(-4 * (wage[:,None] - logit_center_vals)))
    logit5 = 1 / (1 + np.exp(-5 * (wage[:,None] - logit_center_vals)))
    return np.dot(np.hstack([poly_entries, logit1, logit2, logit3, logit4, logit5]), ols_coefs)


# Rewriting this without numpy:
def fitted_differentiable_ls_function(wage, logit_center_vals, ols_coefs, poly_degree):
    logit_center_vals_DM = DM(logit_center_vals)
    poly_entries = wage ** DM(np.arange(poly_degree + 1))
    logit1 = 1 / (1 + exp(-1 * (wage - logit_center_vals_DM)))
    logit2 = 1 / (1 + exp(-2 * (wage - logit_center_vals_DM)))
    logit3 = 1 / (1 + exp(-3 * (wage - logit_center_vals_DM)))
    logit4 = 1 / (1 + exp(-4 * (wage - logit_center_vals_DM)))
    logit5 = 1 / (1 + exp(-5 * (wage - logit_center_vals_DM)))
    return vertcat(poly_entries, logit1, logit2, logit3, logit4, logit5).T @ DM(ols_coefs) # 2.79566189e+00 + -8.56235500e-01 * wage + 7.85001289e-02 * wage ** 2 + -2.01661620e-03 * wage ** 3 # DM(regressors).T @ DM(ols_coefs)


def residual_labor_supply(firm_id, worker_epsilons, other_log_wages, beta):
    log_wages_paying_0 = other_log_wages.copy()
    log_wages_paying_0[firm_id] = 0
    indirect_utility_firm_pays_0 = beta * log_wages_paying_0 + worker_epsilons
    compensating_diff_to_work_at_firm = np.amax(indirect_utility_firm_pays_0, axis=1) - indirect_utility_firm_pays_0[:, firm_id]
    return np.sort(compensating_diff_to_work_at_firm)

def residual_labor_supply_with_habits(firm_id, worker_epsilons, other_log_wages, habit_utilities, beta):
    log_wages_paying_0 = other_log_wages.copy()
    log_wages_paying_0[firm_id] = 0
    indirect_utility_firm_pays_0 = beta * log_wages_paying_0 + worker_epsilons + habit_utilities
    compensating_diff_to_work_at_firm = np.amax(indirect_utility_firm_pays_0, axis=1) - indirect_utility_firm_pays_0[:, firm_id]
    return np.sort(compensating_diff_to_work_at_firm)


def theoretical_ls(own_log_wage, log_wages, firm_id, beta): # Exploits logit preferences
    log_wages_paying_own_log_wage = log_wages.copy()
    log_wages_paying_own_log_wage[firm_id] = own_log_wage # comment out to get the non-oligopsony case where the firm ignores its affect on the aggregate. Approximation errors abound; can get share > 1
    correct_lambda = 1 / np.sum(np.exp(beta * log_wages_paying_own_log_wage))
    choice_prob = correct_lambda * exp(own_log_wage)
    return choice_prob

def profits(wage, log_MRPL, firm_id, logit_center_vals, ols_coefs, poly_degree):
    # quantity_employed = np.searchsorted(ls_func, wage)
    return (exp(log_MRPL[firm_id]) - exp(wage)) * fitted_differentiable_ls_function(wage, logit_center_vals, ols_coefs, poly_degree)
