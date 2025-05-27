# Convert to matrix (Stan requires this format)
library(lavaan) # for the PoliticalDemocracy data
library(blavaan)
data("PoliticalDemocracy")
df <- PoliticalDemocracy
data_matrix <- as.matrix(df)

# Create Stan data list
stan_data <- list(
  N = nrow(df),  # Number of observations
  Y1 = data_matrix[,1:4],# Observed data (75 x 11 matrix)
  Y2 = data_matrix[,5:8], 
  X = data_matrix[,9:11]
)


library(rstan)
# Compile the Stan model
stan_model <- stan_model("stan_sem_politics.stan")

# Run MCMC sampling
fit <- sampling(stan_model, data = stan_data, iter = 2000, chains = 2, warmup = 1000, seed = 123)


posterior_samples <- extract(fit)  # Extracts MCMC samples
summary(fit)$summary 