import numpy as np
from casadi import *
from HelperFunctions import *
import scipy

# Note the importance of assuming the logit shocks are perfectly persistent

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


n_approx_points = 5000
poly_degree = 4
logit_center_vals = np.array([log_MRPL[0] + i for i in np.linspace(-2, 6, 8)])




initial_wage_guess = 9.285701148

# estimates = np.full(n_firms, initial_wage_guess)
estimates = np.empty(n_firms)
log_wages = np.full(n_firms, initial_wage_guess)


# for firm_id in np.arange(n_firms):

#     ls_func_i = residual_labor_supply(firm_id, worker_epsilons, log_wages, beta)

#     # Constructing a differentiable LS function

#     ols_coefs = construct_differentiable_ls_func(ls_func_i, n_approx_points, poly_degree, logit_center_vals)

#     print(f"{fitted_differentiable_ls_function(log_wages[1], logit_center_vals, ols_coefs, poly_degree) = }")
#     print(f"{np.searchsorted(ls_func_i, log_wages[1])/n_workers = }")

#     firm_i_log_wage = SX.sym("firm_i_log_wage")

#     x = firm_i_log_wage

#     x0 = initial_wage_guess - 0.4

#     obj = - profits(firm_i_log_wage, log_MRPL, firm_id, logit_center_vals, ols_coefs, poly_degree)

#     nlp = {
#         "x": x,
#         "f": obj,
#     }


#     solver = nlpsol("solver", "ipopt", nlp, {"ipopt.print_level": 0, "ipopt.tol": 1e-11, 'print_time':0})
#     solution = solver(
#         x0=x0,
#     )

#     print(f"{solution['x'] = }")

#     estimates[firm_id] = solution['x']
#     if firm_id < 10:
#         log_wages = np.full(n_firms, np.mean(estimates[:firm_id + 1]))
#     else:
#         log_wages = np.full(n_firms, np.mean(estimates[5:firm_id + 1]))
    



# print(estimates)

log_wages = np.full(n_firms, np.mean(estimates))

estimates_habit = np.empty(n_firms)


matches = np.argmax(worker_epsilons, axis=1)
mean_match_utility = np.mean(np.amax(worker_epsilons, axis=1))

worker_epsilons = worker_epsilons * 0.8

habit_utilities = np.zeros_like(worker_epsilons)
habit_utilities[np.arange(habit_utilities.shape[0]), matches] = mean_match_utility * 0.2

log_wages = np.full(n_firms, np.mean(estimates) - mean_match_utility * 0.2)


# for firm_id in np.arange(n_firms):

#     ls_func_i_habit = residual_labor_supply_with_habits(firm_id, worker_epsilons, log_wages, habit_utilities, beta)

#     # Constructing a differentiable LS function

#     ols_coefs = construct_differentiable_ls_func(ls_func_i_habit, n_approx_points, poly_degree, logit_center_vals)

#     print(f"{fitted_differentiable_ls_function(log_wages[1], logit_center_vals, ols_coefs, poly_degree) = }")
#     print(f"{np.searchsorted(ls_func_i_habit, log_wages[1])/n_workers = }")

#     print(f"{fitted_differentiable_ls_function(log_wages[1] - 1, logit_center_vals, ols_coefs, poly_degree) = }")
#     print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] + 1)/n_workers = }")
#     print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] + 0.7)/n_workers = }")
#     print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] + 0.5)/n_workers = }")
#     print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] + 0)/n_workers = }")
#     print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] - 0.5)/n_workers = }")
#     print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] - 0.7)/n_workers = }")
#     print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] - 1)/n_workers = }")
#     print(f"{np.searchsorted(ls_func_i, log_wages[1] - 1)/n_workers = }")
#     print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] - 3)/n_workers = }")
#     print(f"{np.searchsorted(ls_func_i, log_wages[1] - 3)/n_workers = }")

#     def kinked_ls_sol(firm_i_log_wage):
#         quantity_employed = np.searchsorted(ls_func_i_habit, firm_i_log_wage)/n_workers
#         return - (exp(log_MRPL[firm_id]) - exp(firm_i_log_wage)) * quantity_employed


#     firm_i_log_wage = SX.sym("firm_i_log_wage")

#     x = firm_i_log_wage

#     x0 = initial_wage_guess - 0.7
    
#     res = scipy.optimize.minimize(kinked_ls_sol, x0, method='Nelder-Mead', tol=1e-6)

#     print(res.x)


#     # obj = - profits(firm_i_log_wage, log_MRPL, firm_id, logit_center_vals, ols_coefs, poly_degree)

#     # nlp = {
#     #     "x": x,
#     #     "f": obj,
#     # }


#     # solver = nlpsol("solver", "ipopt", nlp, {"ipopt.print_level": 0, "ipopt.tol": 1e-11, 'print_time':0})
#     # solution = solver(
#     #     x0=x0,
#     # )

#     # print(f"{solution['x'] = }")

#     # estimates_habit[firm_id] = solution['x']
#     estimates_habit[firm_id] = res.x
#     if firm_id < 10:
#         log_wages = np.full(n_firms, np.mean(estimates_habit[:firm_id + 1]))
#     else:
#         log_wages = np.full(n_firms, np.mean(estimates_habit[5:firm_id + 1]))


# print(estimates_habit)    



log_wages = np.full(n_firms, 8.14) # np.mean(estimates_habit) + 0.1)


for firm_id in np.repeat(np.arange(n_firms), 1):

    ls_func_i_habit = residual_labor_supply_with_habits(firm_id, worker_epsilons, log_wages, habit_utilities, beta)

    # Constructing a differentiable LS function

    ols_coefs = construct_differentiable_ls_func(ls_func_i_habit, n_approx_points, poly_degree, logit_center_vals)

    print(f"{fitted_differentiable_ls_function(log_wages[1], logit_center_vals, ols_coefs, poly_degree) = }")
    print(f"{np.searchsorted(ls_func_i_habit, log_wages[1])/n_workers = }")

    print(f"{fitted_differentiable_ls_function(log_wages[1] - 1, logit_center_vals, ols_coefs, poly_degree) = }")
    print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] + 1.2)/n_workers = }")
    print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] + 1)/n_workers = }")
    print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] + 0.7)/n_workers = }")
    print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] + 0.5)/n_workers = }")
    print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] + 0)/n_workers = }")
    print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] - 0.5)/n_workers = }")
    print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] - 0.7)/n_workers = }")
    print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] - 1)/n_workers = }")
    # print(f"{np.searchsorted(ls_func_i, log_wages[1] - 1)/n_workers = }")
    print(f"{np.searchsorted(ls_func_i_habit, log_wages[1] - 3)/n_workers = }")
    # print(f"{np.searchsorted(ls_func_i, log_wages[1] - 3)/n_workers = }")

    def kinked_ls_sol(firm_i_log_wage):
        quantity_employed = np.searchsorted(ls_func_i_habit, firm_i_log_wage)/n_workers
        return - (exp(log_MRPL[firm_id]) - exp(firm_i_log_wage)) * quantity_employed


    firm_i_log_wage = SX.sym("firm_i_log_wage")

    x = firm_i_log_wage

    x0 = initial_wage_guess - 0.7
    
    res = scipy.optimize.minimize(kinked_ls_sol, x0, method='Nelder-Mead', tol=1e-6)

    print(res.x)


    # obj = - profits(firm_i_log_wage, log_MRPL, firm_id, logit_center_vals, ols_coefs, poly_degree)

    # nlp = {
    #     "x": x,
    #     "f": obj,
    # }


    # solver = nlpsol("solver", "ipopt", nlp, {"ipopt.print_level": 0, "ipopt.tol": 1e-11, 'print_time':0})
    # solution = solver(
    #     x0=x0,
    # )

    # print(f"{solution['x'] = }")

    # estimates_habit[firm_id] = solution['x']
    estimates_habit[firm_id] = res.x
    # log_wages = np.full(n_firms, np.mean(estimates_habit))


print(estimates_habit)    


