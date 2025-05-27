data {
  int<lower=1> N;               // Number of observations
  matrix[N, 3] X;               // Observed indicators for ind60
  matrix[N, 4] Y1;              // Observed indicators for dem60
  matrix[N, 4] Y2;              // Observed indicators for dem65
}

parameters {
  vector[3] lambda_ind60;       // Loadings for ind60
  vector[4] lambda_dem60;       // Loadings for dem60
  vector[4] lambda_dem65;       // Loadings for dem65
  vector[N] ind60;              // Latent variable ind60
  vector[N] dem60;              // Latent variable dem60
  vector[N] dem65;              // Latent variable dem65
  real beta_1;                  // Regression coefficient dem60 ~ ind60
  real beta_2;                  // Regression coefficient dem65 ~ ind60
  real beta_3;                  // Regression coefficient dem65 ~ dem60
  cholesky_factor_cov[8] L_Omega_Y; // Cholesky factor for Y residual covariance structure
  vector<lower=0>[8] sigma_Y;   // Measurement error standard deviations for Y
  cholesky_factor_cov[3] L_Omega_X; // Cholesky factor for X residual covariance structure
  vector<lower=0>[3] sigma_X;   // Measurement error standard deviations for X
  
  // Residual correlations as individual parameters, bounded between -1 and 1
  real<lower=-1, upper=1> rho_Y1;  // Residual correlation y1~~y5
  real<lower=-1, upper=1> rho_Y2;  // Residual correlation y2~~y4
  real<lower=-1, upper=1> rho_Y3;  // Residual correlation y2~~y6
  real<lower=-1, upper=1> rho_Y4;  // Residual correlation y3~~y7
  real<lower=-1, upper=1> rho_Y5;  // Residual correlation y4~~y8
  real<lower=-1, upper=1> rho_Y6;  // Residual correlation y6~~y8
}

transformed parameters {
  matrix[11, 11] Omega;
  matrix[8, 8] Omega_Y;
  Omega = diag_matrix(rep_vector(1, 11)); // Initialize as identity
  
  // Construct block-diagonal covariance matrix
  Omega[1:3, 1:3] = multiply_lower_tri_self_transpose(L_Omega_X);
  
  // Define Omega_Y with specified residual covariances
  Omega_Y = multiply_lower_tri_self_transpose(L_Omega_Y);
  Omega_Y[1, 5] = rho_Y1;
  Omega_Y[2, 4] = rho_Y2;
  Omega_Y[2, 6] = rho_Y3;
  Omega_Y[3, 7] = rho_Y4;
  Omega_Y[4, 8] = rho_Y5;
  Omega_Y[6, 8] = rho_Y6;
  Omega_Y[5, 1] = rho_Y1; // Symmetric entries
  Omega_Y[4, 2] = rho_Y2;
  Omega_Y[6, 2] = rho_Y3;
  Omega_Y[7, 3] = rho_Y4;
  Omega_Y[8, 4] = rho_Y5;
  Omega_Y[8, 6] = rho_Y6;
  
  Omega[4:11, 4:11] = Omega_Y;
}

model {
  // Priors matching Blavaan defaults
  lambda_ind60 ~ normal(1, 2);
  lambda_dem60 ~ normal(1, 2);
  lambda_dem65 ~ normal(1, 2);
  sigma_Y ~ cauchy(0, 2.5);
  sigma_X ~ cauchy(0, 2.5);
  beta_1 ~ normal(0, 1);
  beta_2 ~ normal(0, 1);
  beta_3 ~ normal(0, 1);
  L_Omega_Y ~ lkj_corr_cholesky(2);
  L_Omega_X ~ lkj_corr_cholesky(2);
  
  // Weakly informative priors for residual correlations
  rho_Y1 ~ normal(0, 0.1); 
  rho_Y2 ~ normal(0, 0.1);
  rho_Y3 ~ normal(0, 0.1);
  rho_Y4 ~ normal(0, 0.1);
  rho_Y5 ~ normal(0, 0.1);
  rho_Y6 ~ normal(0, 0.1);
  
  for (n in 1:N) {
    // Measurement model with full multivariate normal structure
    row_vector[11] XY_obs;
    XY_obs[1:3] = X[n];
    XY_obs[4:7] = Y1[n];
    XY_obs[8:11] = Y2[n];
    XY_obs ~ multi_normal(rep_vector(0, 11), quad_form_diag(Omega, append_row(sigma_X, sigma_Y)));
    
    // Structural model
    dem60[n] ~ normal(beta_1 * ind60[n], 1);
    dem65[n] ~ normal(beta_2 * ind60[n] + beta_3 * dem60[n], 1);
  }
}
