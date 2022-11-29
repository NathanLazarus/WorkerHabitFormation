import numpy as np
from casadi import *
from HelperFunctions import *

np.random.seed(10)

n_workers = 400000
n_firms = 20

beta = 1

constant_MRP = 10

markdown = - np.log(beta / (1 + beta))

log_MRPL = np.full(n_firms, constant_MRP)

current_firm = np.repeat(np.arange(n_firms), n_workers / n_firms)

worker_epsilons = np.random.gumbel(loc=0, scale=1, size=n_workers * n_firms).reshape(n_workers, n_firms)

# print(worker_epsilons)

# Claim: roughly admits a Bellman representation with the state variable being the quantity of workers employed by each firm, ignoring the joint distribution of epsilons and employers
# This will be better in steady state (if there are no preference shocks over time), where conditional on number of workers employed their compensating diffs should be above a threshold...but confusing with n_firms epsilon values per worker
# Could also study for a very small number of workers, keeping track of each worker's previous employer. This is like the stylized IO branding model where 2 firms were competing over 1 consumer

log_wages = np.full(n_firms, 9.28568) # log_MRPL - markdown - 0.02

habit_utilities = np.zeros_like(worker_epsilons)
habit_utilities[np.arange(habit_utilities.shape[0]), current_firm] = 2

indirect_utilities = beta * log_wages + worker_epsilons + habit_utilities

ls_func3 = residual_labor_supply(3, worker_epsilons, log_wages, beta)

ls_func3_habit = residual_labor_supply_with_habits(3, worker_epsilons, log_wages, habit_utilities, beta)



# Constructing a differentiable LS function

this_ls_func = ls_func3

approx_start = this_ls_func[int(floor(n_workers * 0.0005))]
approx_end = this_ls_func[int(ceil(n_workers * 0.99))]
# print(approx_start, approx_end)
n_approx_points = 5000
approx_points = np.linspace(approx_start, approx_end, n_approx_points)

approx_points_ls_value = np.searchsorted(this_ls_func, approx_points) / n_workers

poly_degree = 4

poly_entries = approx_points[:,None] ** np.arange(poly_degree + 1)

# logit_center_vals = 0
logit_center_vals = np.array([log_MRPL[3] + i for i in np.linspace(-2, 6, 8)])
logit1 = 1 / (1 + np.exp(-1 * (approx_points[:,None] - logit_center_vals)))
logit2 = 1 / (1 + np.exp(-2 * (approx_points[:,None] - logit_center_vals)))
logit3 = 1 / (1 + np.exp(-3 * (approx_points[:,None] - logit_center_vals)))
logit4 = 1 / (1 + np.exp(-4 * (approx_points[:,None] - logit_center_vals)))
logit5 = 1 / (1 + np.exp(-5 * (approx_points[:,None] - logit_center_vals)))
approx_points_dictionary = np.hstack([poly_entries, logit1, logit2, logit3, logit4, logit5])
# approx_points_dictionary = poly_entries

[ols_coefs, resids, rank, s] = np.linalg.lstsq(approx_points_dictionary, approx_points_ls_value, rcond=-1)

# print(ols_coefs)
# print(resids)
# print(f"{rank = }, {np.shape(approx_points_dictionary) = }")
# print(np.shape(ols_coefs))

# print(f"{fitted_differentiable_ls_function_numpy(np.array([9]), logit_center_vals, ols_coefs) = }")
print(f"{fitted_differentiable_ls_function(9, logit_center_vals, ols_coefs, poly_degree) = }")
print(f"{np.searchsorted(this_ls_func, 9)/n_workers = }")





firm3_log_wage = SX.sym("firm3_log_wage")

x = firm3_log_wage

x0 = 8 # log_MRPL[3] - markdown + 0.005

obj = - profits(firm3_log_wage, log_MRPL, 3, logit_center_vals, ols_coefs, poly_degree)

nlp = {
    "x": x,
    "f": obj,
    # "g": constraint,
}


print(f"{profits(9, log_MRPL, 3, logit_center_vals, ols_coefs, poly_degree) = }")
print(f"{profits(9.5, log_MRPL, 3, logit_center_vals, ols_coefs, poly_degree) = }")
print(f"{profits(9.8, log_MRPL, 3, logit_center_vals, ols_coefs, poly_degree) = }")
print(f"{profits(10.2, log_MRPL, 3, logit_center_vals, ols_coefs, poly_degree) = }")
print(f"{profits(12, log_MRPL, 3, logit_center_vals, ols_coefs, poly_degree) = }")

myguess = log_MRPL[3] - markdown
print(log_MRPL[3] - markdown)
print(log_MRPL[3])
print(profits(myguess, log_MRPL, 3, logit_center_vals, ols_coefs, poly_degree))

# lower_bounds_x = 7
# upper_bounds_x = log_MRPL[3]

solver = nlpsol("solver", "ipopt", nlp,) #  {"ipopt.print_level": 0, "ipopt.tol": 1e-11, 'print_time':0})
solution = solver(
    x0=x0,
    # lbg=lower_bounds_g,
    # ubg=upper_bounds_g,
    # lbx=lower_bounds_x,
    # ubx=upper_bounds_x,
)

print(solution)

print(f"firm 3 profit: {solution['f']}")
print(f"firm 3 wage: {exp(solution['x'])}")
print(f"firm 3 log wage: {solution['x']}")
print(f"firm 3 quantity: {np.searchsorted(this_ls_func, solution['x'])}")
print(f"firm 3 quantity from fitted LS: {fitted_differentiable_ls_function(solution['x'], logit_center_vals, ols_coefs, poly_degree) * n_workers}")
print(f"firm 3 MRPL: {exp(log_MRPL[3])}")


print("true profits")
print(np.searchsorted(this_ls_func, solution['x']) * (exp(log_MRPL[3]) - exp(solution['x'])))
print("addding 0.003")
print(np.searchsorted(this_ls_func, solution['x'] + 0.003) * (exp(log_MRPL[3]) - exp(solution['x'] + 0.003)))
print("subtracting 0.003")
print(np.searchsorted(this_ls_func, solution['x'] - 0.003) * (exp(log_MRPL[3]) - exp(solution['x'] - 0.003)))

print("my guess")
print(myguess)
print(np.searchsorted(this_ls_func, myguess) * (exp(log_MRPL[3]) - exp(myguess)))
print(f"{solution['x'] = }")
print(np.searchsorted(this_ls_func, solution['x']) * (exp(log_MRPL[3]) - exp(solution['x'])))



# def theoretical_profits(log_wage, other_log_wages, log_MRPL, firm_id, beta):
#     return theoretical_ls(log_wage, other_log_wages, firm_id, beta) * (exp(log_MRPL[firm_id]) - exp(log_wage))

# print(theoretical_ls(solution['x'], log_wages, 3, beta))

# print(f"theoretical profits solution: {theoretical_profits(solution['x'], log_wages, log_MRPL, 3, beta)}")
# print(f"theoretical profits myguess: {theoretical_profits(myguess, log_wages, log_MRPL, 3, beta)}")

# for i in np.linspace(9.7235, 9.7245, 30):
#     print(f"")
#     print(f"{i = }, {theoretical_profits(i, log_wages, log_MRPL, 3, beta) = }")

print(f"{solution['x'] = }")
print(log_wages)

# Theoretical (exact) solution: 9.72423 (different from "myguess", based on the markdown, because of oligopsony: the firm internalizing its effect on the denominator)
# Interpolated solution: 9.72348

# Monopolistic competition would give a markdown of 50% here, so MRPL - wage = log(2)
# The symmetric oligopsony instead gives a markdown of 51.05%, so MRPL - wage = log(2.043)