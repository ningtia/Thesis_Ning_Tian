// nonlinear_sv.stan
// Nonlinear stochastic volatility with a small neural network transition
// Inputs: observable lagged returns (y+, y-) and exogenous covariates
// Estimation: NUTS (Hamiltonian Monte Carlo)

data {
    int<lower=2> T;                         // number of time periods
    int<lower=1> D;                         // dimension of exogenous covariates x
    int<lower=1> K;                         // number of hidden units
    vector[T] y;                            // observed log returns (percent)
    matrix[T-1, D] X;                       // exogenous covariates x_{t-1}, rows t=2..T
    int<lower=0, upper=1> use_student_t;    // 0 = Gaussian, 1 = Student-t
 }

 transformed data {
    // Pre-compute positive and negative return parts
    vector[T-1] y_pos;                      // y_{t-1}^+ for t = 2..T
    vector[T-1] y_neg;                      // y_{t-1}^- for t = 2..T
    int D_nn = D + 3;                       // NN input dim: h_{t-1}, y+, y-, x_{t-1}

    for (i in 1:(T-1)) {
        y_pos[i]= fmax(y[i], 0.0);
        y_neg[i]= fmin(y[i], 0.0);
    }
 }

 parameters{
     //---SV parameters--

     real mu;                               // log-variance mean
     real phi_raw;                          // raw persistence (unconstrained)
     real<lower=0> sigma_eta;               // volatility of log-variance

     //---NN parameters (one hidden layer)--
     matrix[K,D_nn] W1;                     // input-to-hidden weights
     vector[K] b1;                          // hidden biases
     vector[K] w2;                          // hidden-to-output weights
     real b2;                               // output bias
     real<lower=0> s;                       // output bound:|g| <= s

     //---Hierarchical weight scale--
     real<lower=0> tau_w;                   // shared scale for NN weights

     //---Latent path (non-centered)--
     vector[T] eta_raw;                     // auxiliary standard normals

     //---Heavy tails (optional)--
     real<lower=0> nu_minus2[use_student_t ? 1:0];
}


transformed parameters {
     real phi = tanh(phi_raw);              // persistence in (-1,1)
     vector[T] h;                           // latent log-variance path

     //Reconstruct h_1 (stationary distribution)
     h[1] = mu + (sigma_eta / sqrt(1 - phi ^ 2)) *eta_raw[1];
 
     //Reconstruct h_2,...,h_T via nonlinear transition
     for(t in 2:T){
        //Assemble NN input vector: [h_{t-1}, y_{t-1}^+, y_{t-1}^-, x_{t-1}]
        vector[D_nn] z;
        z[1] = h[t-1];
        z[2] = y_pos[t-1];
        z[3] = y_neg[t-1];
        z[4:D_nn] = X[t-1]’;

        //Forward pass: one hidden layer + bounded output
        vector[K] a = tanh(W1 * z + b1);
        real nn_out = s * tanh(dot_product(w2, a) + b2);

        //Transition equation(non-centered)
        h[t] = mu +phi * (h[t-1] - mu) + nn_out + sigma_eta * eta_raw[t];
    }
}

model{
    //=== PRIORS ===

    //SV parameters
    mu ~ normal(0,5);
    phi_raw ~ normal(2, 1);     // priorcentersphinear0.96
    sigma_eta ~ normal(0,1);    //half-normal(declared <lower=0>)

    //NN weights: hierarchical regularization
    tau_w ~ normal(0,1);    //half-normal
    to_vector(W1) ~ normal(0, tau_w);
    b1 ~ normal(0,1);
    w2 ~ normal(0,tau_w);
    b2 ~ normal(0,1);
    s ~ normal(0, 1);       // half-normal
    
    //Latent auxiliary variables
    eta_raw ~ std_normal();
    
    //Heavy tails (if enabled)
    if(use_student_t){
    nu_minus2 ~ gamma(2, 0.1); //nu=nu_minus2+2 >2
    }
    
    //=== LIKELIHOOD===
    if(use_student_t){
        real nu = nu_minus2[1] + 2;
        for (t in 1:T){
            y[t] ~ student_t(nu, 0, exp(h[t] / 2));
        }
    } else {
        for (t in 1:T){
            y[t] ~ normal(0, exp(h[t] / 2));
        }
    }
}



generated quantities{
    //Posterior predictive log-likelihood (for LOO-CV)
    vector[T] log_lik;

    //One-step-ahead volatility forecast (h_{T+1} | data)
    real h_forecast;
    real vol_forecast;

    //Log-likelihood computation
    if(use_student_t){
        real nu = nu_minus2[1] + 2;
        for (t in 1:T)
            log_lik[t] = student_t_lpdf(y[t] | nu, 0, exp(h[t] / 2));
    } else {
        for (t in 1:T)
            log_lik[t] = normal_lpdf(y[t] | 0, exp(h[t] / 2));
    }

    //Forecast:drawh_{T+1}
    {
        vector[D_nn] z_T;
        z_T[1] = h[T];
        z_T[2] = fmax(y[T], 0.0);
        z_T[3] = fmin(y[T], 0.0);
        z_T[4:D_nn] = X[T]; // I add it

        //Note: x_T must be passed separately for true forecasting;
        //here we use the last available row for illustration
        //In practice,supply x_T as additional data
        // for (d in 1:D) z_T[3+d] = X[T-1, d];    // placeholder
        
        vector[K] a_T = tanh(W1 * z_T + b1);
        real nn_T = s * tanh(dot_product(w2, a_T) + b2);
        h_forecast = mu + phi * (h[T]- mu) + nn_T + sigma_eta * normal_rng(0, 1);
        vol_forecast = exp(h_forecast / 2);
    }
}