pacman::p_load(data.table, foreach)

n_firms = 10
n_workers = 1000000
beta = 1 # elasticity of LS is beta w / (w - b), b is 0, so just beta. Therefore markdown is beta / (1 + beta)

set.seed(5)

worker_epsilons = matrix(EnvStats::revd(n_firms * n_workers), nrow = n_workers, ncol = n_firms)

log_wages = seq(9.6, 10.5, length.out = 10)
log_MRPLs = log_wages + log(2)

indirect_utilities = beta * matrix(rep(log_wages, each = n_workers), nrow = n_workers) + worker_epsilons

indirect_utility_firm_1_pays_0 = beta * matrix(rep(c(0, log_wages[2:n_firms]), each = n_workers), nrow = n_workers) + worker_epsilons
favorite_firm_utility = matrixStats::rowMaxs(indirect_utility_firm_1_pays_0)
compensating_diff_to_work_at_firm_1 = favorite_firm_utility - worker_epsilons[, 1]

indirect_utility_firm_7_pays_0 = beta * matrix(rep(c(log_wages[1:6], 0, log_wages[8:n_firms]), each = n_workers), nrow = n_workers) + worker_epsilons
favorite_firm_utility = matrixStats::rowMaxs(indirect_utility_firm_7_pays_0)
compensating_diff_to_work_at_firm_7 = favorite_firm_utility - worker_epsilons[, 7]

asdf = data.table(indirect_utilities)
asdf[, index := .I]
asdf[, maxCol := which.max(.SD), by = index]

empirical_choice_probs = table(asdf$maxCol) / nrow(asdf)

theoretical_choice_probs = exp(beta * log_wages) / sum(exp(beta * log_wages))

lambda = 1 / sum(exp(beta * log_wages))

rough_theoretical_choice_probs = function(own_log_wage) {
  lambda * exp(own_log_wage)
}

empirical_choice_probs
theoretical_choice_probs

# firm_demand perfectly elastic. Should generalize to elasticity of demand theta

log_MRPL_7 = log_MRPLs[7]

# Oligopsony: here the firm's taking into account its effect on lambda

profit_table = foreach(log_wage = seq(10.1, 10.3, by = 0.001), .combine = rbind) %do% {
  quantity_employed = sum(compensating_diff_to_work_at_firm_7 < log_wage)
  profits = (exp(log_MRPL_7) - exp(log_wage)) * quantity_employed
  data.table(log_wage = log_wage, quantity_employed = quantity_employed, profits = profits)
}

profit_table
profit_table[which.max(profit_table$profits)]

# This is the limit case with large n_firms

profit_table_theoretical = foreach(log_wage = seq(10.18, 10.22, by = 0.0001), .combine = rbind) %do% {
  quantity_employed = rough_theoretical_choice_probs(log_wage) * n_workers
  profits = (exp(log_MRPL_7) - exp(log_wage)) * quantity_employed
  data.table(log_wage = log_wage, quantity_employed = quantity_employed, profits = profits)
}

profit_table_theoretical[which.max(profit_table_theoretical$profits)]


