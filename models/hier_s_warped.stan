functions {
  void add_iter();
  int get_iter();
  real generalized_inverse_gaussian_lpdf(real x, int p,real a, real b) {
    return p*0.5*log(a / b)- log(2*modified_bessel_second_kind(p, sqrt(a*b)))+
     (p - 1)*log(x)- (a*x + b / x)*0.5;
     }

  vector gp_pred_rng(array[] vector x2,
                     vector y1, array[] vector x1,
                     real alpha, array[] real rho, real sigma, real delta) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2;
    {
      matrix[N1, N1] K =   gp_exp_quad_cov(x1, alpha, rho)
                         + diag_matrix(rep_vector(square(sigma), N1));
      matrix[N1, N1] L_K = cholesky_decompose(K);

      vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
      vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
      matrix[N1, N2] k_x1_x2 = gp_exp_quad_cov(x1, x2, alpha, rho);
      vector[N2] f2_mu = (k_x1_x2' * K_div_y1);
      matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      matrix[N2, N2] cov_f2 =   gp_exp_quad_cov(x2, alpha, rho) - v_pred' * v_pred
                              + diag_matrix(rep_vector(delta, N2));
      f2 = multi_normal_rng(f2_mu, cov_f2);
    }
    return f2;
  }
  vector softplus(vector x) {
    return log1p(exp(x));
  }

  matrix qp_rational_quadratic(array[] vector x, real alpha, array[] real rho, real strength) {
    int N = size(x);
    matrix[N,N] out;
   for (i in 1:N) {
      for (j in i:N) {
         real diff_x = (x[i][1]-x[j][1])^2/(2*strength*rho[1]^2);
         real diff_y = (x[i][2]-x[j][2])^2/(2*strength*rho[2]^2);
         real val = alpha^2*(1+diff_x+diff_y)^(-strength);
         out[i,j] = val;
         out[j,i] = val;
      }
   }
   return out;
  }

  matrix matern_five_halves(array[] vector x, real alpha, real rho) {
    int N = size(x);
    matrix[N,N] out;
    for (i in 1:N) {
      for (j in i:N) {
         real dist = distance(x[i],x[j]);
         real val = alpha^2*(1.0+sqrt(5)/rho*dist+5.0/(3.0*square(rho))*square(dist))*exp(-sqrt(5)/rho*dist);
         out[i,j] = val;
         out[j,i] = val;
      }
   }
   return out;
  }

  matrix matern_three_halves(array[] vector x, real alpha, real rho) {
    int N = size(x);
    matrix[N,N] out;
    for (i in 1:N) {
      for (j in i:N) {
         real dist = distance(x[i],x[j]);
         real val = alpha^2*(1.0+sqrt(3)/rho*dist)*exp(-sqrt(3)/rho*dist);
         out[i,j] = val;
         out[j,i] = val;
      }
   }
   return out;
  }

  vector rep_each(vector x, int K) {
    int N = rows(x);
    vector[N * K] y;
    int pos = 1;
    for (n in 1:N) {
      for (k in 1:K) {
        y[pos] = x[n];
        pos += 1;
      }
    }
    return y;
  }
  real map_interval(real x) {
    if (x<0) {
      return 0;
    } else if (x>1) {
      return 1;
    } else {
      return x;
    }
  }

  vector warp(vector x,vector center, real a, real b, real reach, real direction) {
      vector[2] out;
      real norm = distance(x,center);
      real r = norm*(direction*beta_cdf(map_interval(1-(norm*reach)) | a,b)+1);
      real theta = atan2(x[2],x[1]);
      out[1] = r*cos(theta);
      out[2] = r*sin(theta);
      return out+center;
  }

      real partial_sum(array[] real y_slice,
                   int start, int end,
                   int N_rad,
                   real nu,
                   int t_len,
                   vector radius_acc,
                   array[] int idents,
                   vector amplitude_values,
                   array[] int location_identifier,
                   array[] int identifier,
                   vector t,
                   vector F_phi,vector sigma_model) {


    return student_t_lpdf(y_slice | nu,
    (rep_each(radius_acc[idents],t_len)^2)[start:end].*amplitude_values[location_identifier[start:end]].*sin(2*pi()*0.05*t[start:end]-F_phi[location_identifier[start:end]]),sigma_model[location_identifier[start:end]]);
  }
}



data {

  int<lower=1> N_locations;

  array[N_locations] int loc_ids_1;
  array[N_locations] int loc_ids_2;
  int iters;
  int N_cells;
  int<lower=1> N; // num series (all combined)
  int<lower=1> total; // total number of datapoints
  int<lower=1> D;
  int<lower=1> N_rad;
  real<lower=0> tau;
  array[N] vector[D] x;
  //matrix[iters,N] F_V;
  vector[iters] F_V;
  vector[total] t;
  array[D] vector[N_cells] cell_centers;
  array[N_locations] int loc_ids_cell_1;
  array[N_locations] int loc_ids_cell_2;
  //real y[N];
  vector[N] radius;
  array[N_rad] int idents;
  array[total] real displacement;
  array[total] int location_identifier;
  array[total] int identifier;
  array[N] int loc_repeat;
}
transformed data {
  int t_len = 1;
  for (i in 2:total) {
     if(t[i]==0.0) {
       break;
     }
    t_len += 1;
  } 
  int N_unique = total%/%t_len;
}

parameters {

  // material level GP params
  real<lower=0> rho_g;
  real<lower=0> rho_phi;
  real alpha_g;
  real alpha_phi;

  // material mean
  real<lower=0> offset_g;
  real<lower=0> offset_g_std;
  vector[N_locations] offset_g_z;
  real offset_phi;
  real<lower=0> offset_phi_std;
  vector[N_locations] offset_phi_z;

  array[D] vector<lower=0>[N_locations] rho_g_locations;
  array[D] vector<lower=0>[N_locations] rho_phi_locations;  

  // assuming same sigma for each location
  real<lower=0> sigma;
  real<lower=0> sigma2;
  

  vector[N] eta_g;
  vector[N] eta_phi;

  real<lower=1> nu;

  real<lower=0> sigma_model_mu;
  real<lower=0> sigma_model_sigma;
  //vector<lower=0>[N] sigma_model;
  vector<lower=0>[N] sigma_model;


  // handle radius as uncertain value
  vector<lower=0>[N] radius_acc;
  real<lower=0> mu_radius;
  real<lower=0> sigma_radius;


  vector[N_locations] z_alpha_g;
  vector[N_locations] z_alpha_phi;

  real<lower=0> alpha_g_sigma;
  real<lower=0> alpha_phi_sigma;
  real<lower=0> rho_g_sigma;
  real<lower=0> rho_phi_sigma;
}
transformed parameters {
  vector[N] F_g;
  vector[N] F_phi;
  vector[N] amplitude_values;


  vector<lower=0>[N_locations] alpha_g_locations = softplus(alpha_g+z_alpha_g*alpha_g_sigma)+1e-5;
  vector<lower=0>[N_locations] alpha_phi_locations = softplus(alpha_phi+z_alpha_phi*alpha_phi_sigma)+1e-5;

  vector[N_locations] offset_g_locations = offset_g+offset_g_z*offset_g_std;
  vector[N_locations] offset_phi_locations = offset_phi+offset_phi_z*offset_phi_std;
  {
    matrix[N,N] L_K;
    matrix[N,N] L_K2;
    matrix[N,N] K = rep_matrix(0,N,N);
    matrix[N,N] K2 = rep_matrix(0,N,N);
    for (i in 1:N_locations) {
      int start = loc_ids_1[i];
      int end = loc_ids_2[i];
      int cell_start = loc_ids_cell_1[i];
      int cell_end = loc_ids_cell_2[i];



      // local effect
      K[start:end,start:end] = gp_exp_quad_cov(x[start:end], alpha_g_locations[i], rho_g_locations[,i]);
      K2[start:end,start:end] = gp_exp_quad_cov(x[start:end], alpha_phi_locations[i], rho_phi_locations[,i]);

    }
    
    real sq_sigma = square(sigma);
    real sq_sigma2 = square(sigma2);

    // diagonal elements
    for (n in 1:N) {
      K[n, n] = K[n, n] + sq_sigma+1e-5;
      K2[n, n] = K2[n, n] + sq_sigma2+1e-5;
    }

    L_K = cholesky_decompose(K);
    L_K2 = cholesky_decompose(K2);

    F_g =  softplus(L_K*eta_g+offset_g_locations[loc_repeat]);
    //amplitude_values = 1e-6*(2*to_vector(F_V[get_iter(),])) ./ (9*F_g);
    amplitude_values = 1e-6*(2*F_V[get_iter()]) ./ (9*F_g);

    F_phi =  asin(inv_logit(L_K2*eta_phi+offset_phi_locations[loc_repeat]));
    //F_phi =  (asin(sin(L_K2*eta_phi+offset_phi_locations[loc_repeat]))+pi()/2-0.1)/1.9;
  }
}

model {
  // global effects
  rho_g ~ std_normal();
  rho_phi ~ std_normal();
  
  alpha_g ~ std_normal();
  alpha_phi ~ normal(0,0.5);

  offset_g ~ normal(50,15); // 30 10
  offset_phi ~ normal(0,0.1);
  offset_g_std ~ std_normal();
  offset_phi_std ~ std_normal();
  offset_g_z ~ std_normal();
  offset_phi_z ~ std_normal();

  // local effects
  for (i in 1:N_locations) {
    for (k in 1:D) {
      rho_g_locations[k][i] ~ generalized_inverse_gaussian(2,15,rho_g_sigma); // -1 ,5
      rho_phi_locations[k][i] ~ generalized_inverse_gaussian(2,15,rho_phi_sigma); // 2, 15
    }
    z_alpha_g[i] ~ std_normal();
    z_alpha_phi[i] ~ normal(0,0.5);
  }

  eta_g ~ std_normal();
  eta_phi ~ std_normal();

  alpha_g_sigma ~ std_normal();
  alpha_phi_sigma ~ std_normal();

  rho_g_sigma ~ std_normal();
  rho_phi_sigma ~ std_normal();

  nu ~ gamma(2,0.1);
  
  sigma ~ normal(0,0.1);
  sigma2 ~ normal(0,0.1);
  sigma_model_mu ~ std_normal();
  sigma_model_sigma ~ std_normal();
  sigma_model ~ inv_gamma(sigma_model_mu,sigma_model_sigma);

  // radius stuff
  mu_radius ~ normal(6,1);
  sigma_radius ~ inv_gamma(2,0.5);
  radius_acc ~ normal(mu_radius,sigma_radius);
  radius ~ normal(radius_acc,tau);
  
  
  target += reduce_sum(partial_sum, displacement,
                  1,N_rad,nu,t_len,radius_acc,
                  idents,amplitude_values,
                  location_identifier,identifier,t,F_phi,
                  sigma_model);
}

generated quantities {
  vector[N_unique] log_likelihood;
  matrix[N_unique,t_len] y_hat;
  for (i in 0:(N_unique-1)) {
    int start = i*t_len+1;
    int end = (i+1)*t_len;
    y_hat[i+1,] = to_row_vector(student_t_rng(nu,(rep_each(radius_acc[idents],t_len)^2)[start:end].*amplitude_values[location_identifier[start:end]].*sin(2*pi()*0.05*t[start:end]-F_phi[location_identifier[start:end]]),sigma_model[identifier[start:end]]));
    log_likelihood[i+1] = student_t_lpdf(displacement[start:end] | nu, (rep_each(radius_acc[idents],t_len)^2)[start:end].*amplitude_values[location_identifier[start:end]].*sin(2*pi()*0.05*t[start:end]-F_phi[location_identifier[start:end]]),sigma_model[identifier[start:end]]);
  }
  add_iter();
}
