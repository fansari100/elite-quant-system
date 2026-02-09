# Elite Quant System - Advanced Statistical Analysis in R
# Renaissance Technologies-style statistical modeling
# R is favored for risk assessment, forecasting, and quantitative modeling

library(tidyverse)
library(zoo)
library(xts)
library(quantmod)
library(PerformanceAnalytics)
library(rmgarch)
library(rugarch)
library(forecast)
library(tseries)
library(fGarch)
library(copula)
library(MASS)
library(glmnet)
library(caret)
library(jsonlite)

#' =============================================================================
#' REGIME DETECTION - Hidden Markov Model for Market States
#' =============================================================================

detect_market_regimes <- function(returns, n_states = 3) {
  #' Detect market regimes using Hidden Markov Models
  #' Used by Renaissance/Two Sigma for regime-aware trading
  
  library(depmixS4)
  
  # Prepare data
  data <- data.frame(returns = as.numeric(returns))
  data <- na.omit(data)
  
  # Fit HMM with Gaussian emissions
  hmm_model <- depmix(
    returns ~ 1,
    data = data,
    nstates = n_states,
    family = gaussian()
  )
  
  fitted_model <- fit(hmm_model, verbose = FALSE)
  
  # Extract regime probabilities
  posterior_probs <- posterior(fitted_model)
  
  # Get regime characteristics
  regime_stats <- data.frame(
    regime = 1:n_states,
    mean = sapply(1:n_states, function(i) {
      mean(data$returns[posterior_probs$state == i], na.rm = TRUE)
    }),
    volatility = sapply(1:n_states, function(i) {
      sd(data$returns[posterior_probs$state == i], na.rm = TRUE)
    })
  )
  
  list(
    states = posterior_probs$state,
    probabilities = posterior_probs[, 2:(n_states + 1)],
    regime_stats = regime_stats,
    transition_matrix = fitted_model@transition
  )
}

#' =============================================================================
#' DYNAMIC COPULA - Tail Dependency Modeling
#' =============================================================================

fit_dynamic_copula <- function(returns_matrix, copula_type = "t") {
  #' Fit dynamic copula for tail dependency estimation
  #' Critical for portfolio risk in extreme market conditions
  
  # Convert to uniform margins using empirical CDF
  u_data <- apply(returns_matrix, 2, function(x) {
    ecdf(x)(x)
  })
  
  # Handle boundary values
  u_data[u_data == 0] <- 0.001
  u_data[u_data == 1] <- 0.999
  
  # Fit copula based on type
  if (copula_type == "t") {
    cop <- tCopula(dim = ncol(returns_matrix), dispstr = "un")
  } else if (copula_type == "clayton") {
    cop <- claytonCopula(dim = ncol(returns_matrix))
  } else {
    cop <- gumbelCopula(dim = ncol(returns_matrix))
  }
  
  fitted_cop <- fitCopula(cop, u_data, method = "ml")
  
  # Calculate tail dependence
  if (copula_type == "t") {
    nu <- coef(fitted_cop)["df"]
    rho <- coef(fitted_cop)["rho.1"]
    # Lower tail dependence for t-copula
    lambda_l <- 2 * pt(-sqrt((nu + 1) * (1 - rho) / (1 + rho)), df = nu + 1)
    tail_dep <- c(lower = lambda_l, upper = lambda_l)
  } else {
    tail_dep <- c(lower = NA, upper = NA)
  }
  
  list(
    copula = fitted_cop,
    parameters = coef(fitted_cop),
    tail_dependence = tail_dep,
    AIC = AIC(fitted_cop)
  )
}

#' =============================================================================
#' DCC-GARCH - Dynamic Conditional Correlation
#' =============================================================================

fit_dcc_garch <- function(returns_matrix, garch_order = c(1, 1)) {
  #' Fit DCC-GARCH for time-varying correlation estimation
  #' Essential for dynamic portfolio optimization
  
  # Univariate GARCH specification
  uspec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = garch_order),
    mean.model = list(armaOrder = c(0, 0)),
    distribution.model = "std"  # Student-t for fat tails
  )
  
  # Multivariate specification
  mspec <- multispec(replicate(ncol(returns_matrix), uspec))
  
  # DCC specification
  dcc_spec <- dccspec(
    uspec = mspec,
    dccOrder = c(1, 1),
    distribution = "mvt"
  )
  
  # Fit model
  dcc_fit <- dccfit(dcc_spec, data = returns_matrix)
  
  # Extract time-varying correlations
  R_t <- rcor(dcc_fit)
  H_t <- rcov(dcc_fit)
  
  list(
    fit = dcc_fit,
    correlations = R_t,
    covariances = H_t,
    volatilities = sigma(dcc_fit),
    residuals = residuals(dcc_fit, standardize = TRUE)
  )
}

#' =============================================================================
#' ROBUST COVARIANCE ESTIMATION - Ledoit-Wolf Shrinkage
#' =============================================================================

ledoit_wolf_shrinkage <- function(returns_matrix) {
  #' Ledoit-Wolf shrinkage estimator for covariance
  #' Addresses estimation error in high-dimensional settings
  
  n <- nrow(returns_matrix)
  p <- ncol(returns_matrix)
  
  # Sample covariance
  S <- cov(returns_matrix)
  
  # Shrinkage target: scaled identity
  mu <- sum(diag(S)) / p
  F <- mu * diag(p)
  
  # Frobenius norm calculations
  d2 <- sum((S - F)^2) / p
  
  # Calculate optimal shrinkage intensity
  X <- scale(returns_matrix, center = TRUE, scale = FALSE)
  sum_sq <- 0
  for (k in 1:n) {
    xk <- X[k, ]
    sum_sq <- sum_sq + sum((outer(xk, xk) - S)^2)
  }
  b2 <- sum_sq / (n^2 * p)
  
  # Shrinkage intensity
  delta <- max(0, min(1, b2 / d2))
  
  # Shrunk covariance
  Sigma_shrunk <- delta * F + (1 - delta) * S
  
  list(
    covariance = Sigma_shrunk,
    shrinkage_intensity = delta,
    sample_covariance = S
  )
}

#' =============================================================================
#' FACTOR MODEL - PCA-based Factor Analysis
#' =============================================================================

estimate_factor_model <- function(returns_matrix, n_factors = NULL) {
  #' Estimate statistical factor model using PCA
  #' Foundation of Barra-style risk models
  
  # Standardize returns
  returns_std <- scale(returns_matrix)
  
  # PCA decomposition
  pca_result <- prcomp(returns_std, center = FALSE, scale. = FALSE)
  
  # Determine number of factors if not specified
  if (is.null(n_factors)) {
    # Use Kaiser criterion
    eigenvalues <- pca_result$sdev^2
    n_factors <- sum(eigenvalues > 1)
  }
  
  # Factor loadings (betas)
  loadings <- pca_result$rotation[, 1:n_factors]
  
  # Factor returns
  factor_returns <- pca_result$x[, 1:n_factors]
  
  # Variance explained
  var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
  cumulative_var <- cumsum(var_explained)
  
  # Idiosyncratic variance
  fitted <- factor_returns %*% t(loadings)
  residuals <- returns_std - fitted
  idio_var <- apply(residuals, 2, var)
  
  list(
    loadings = loadings,
    factor_returns = factor_returns,
    variance_explained = var_explained[1:n_factors],
    cumulative_variance = cumulative_var[n_factors],
    idiosyncratic_variance = idio_var,
    n_factors = n_factors
  )
}

#' =============================================================================
#' ELASTIC NET ALPHA - Regularized Factor Selection
#' =============================================================================

elastic_net_alpha <- function(X, y, alpha = 0.5, n_folds = 5) {
  #' Elastic Net for alpha factor selection
  #' Combines L1 (sparsity) and L2 (grouping) regularization
  
  # Cross-validated fit
  cv_fit <- cv.glmnet(
    x = as.matrix(X),
    y = y,
    alpha = alpha,
    nfolds = n_folds,
    type.measure = "mse",
    standardize = TRUE
  )
  
  # Get coefficients at optimal lambda
  best_lambda <- cv_fit$lambda.1se  # 1 SE rule for regularization
  coefficients <- coef(cv_fit, s = best_lambda)
  
  # Non-zero factors
  selected_factors <- rownames(coefficients)[which(coefficients != 0)]
  selected_factors <- selected_factors[selected_factors != "(Intercept)"]
  
  # In-sample predictions
  predictions <- predict(cv_fit, newx = as.matrix(X), s = best_lambda)
  
  # Information Coefficient
  ic <- cor(predictions, y, method = "spearman")
  
  list(
    model = cv_fit,
    coefficients = as.vector(coefficients),
    selected_factors = selected_factors,
    n_selected = length(selected_factors),
    lambda = best_lambda,
    information_coefficient = ic,
    r_squared = 1 - cv_fit$cvm[which(cv_fit$lambda == best_lambda)] / var(y)
  )
}

#' =============================================================================
#' RISK METRICS - VaR and Expected Shortfall
#' =============================================================================

calculate_risk_metrics <- function(returns, confidence_levels = c(0.95, 0.99)) {
  #' Calculate comprehensive risk metrics
  #' Including VaR, ES, and various risk ratios
  
  returns <- as.numeric(returns)
  returns <- returns[!is.na(returns)]
  
  # Basic statistics
  mu <- mean(returns)
  sigma <- sd(returns)
  skew <- moments::skewness(returns)
  kurt <- moments::kurtosis(returns)
  
  # VaR and ES calculations
  var_results <- sapply(confidence_levels, function(cl) {
    # Historical VaR
    hist_var <- quantile(returns, 1 - cl)
    
    # Parametric VaR (normal)
    param_var <- mu - sigma * qnorm(cl)
    
    # Cornish-Fisher VaR (skew/kurtosis adjusted)
    z <- qnorm(cl)
    cf_z <- z + (z^2 - 1) * skew / 6 + 
            (z^3 - 3*z) * (kurt - 3) / 24 - 
            (2*z^3 - 5*z) * skew^2 / 36
    cf_var <- mu - sigma * cf_z
    
    # Expected Shortfall (CVaR)
    es <- mean(returns[returns <= hist_var])
    
    c(historical_var = hist_var, parametric_var = param_var, 
      cornish_fisher_var = cf_var, expected_shortfall = es)
  })
  
  colnames(var_results) <- paste0("CL_", confidence_levels * 100)
  
  # Additional risk ratios
  risk_ratios <- c(
    sharpe = mu / sigma * sqrt(252),
    sortino = mu / sd(returns[returns < 0]) * sqrt(252),
    calmar = mu * 252 / abs(min(cumsum(returns))),
    omega = sum(returns[returns > 0]) / abs(sum(returns[returns < 0]))
  )
  
  list(
    var_es = var_results,
    risk_ratios = risk_ratios,
    statistics = c(mean = mu, sd = sigma, skewness = skew, kurtosis = kurt),
    max_drawdown = maxDrawdown(returns)
  )
}

#' =============================================================================
#' COINTEGRATION TESTING - Pairs Trading Foundation
#' =============================================================================

test_cointegration <- function(series1, series2) {
  #' Test for cointegration between two price series
  #' Foundation for statistical arbitrage / pairs trading
  
  # Align series
  n <- min(length(series1), length(series2))
  s1 <- series1[1:n]
  s2 <- series2[1:n]
  
  # Engle-Granger two-step method
  # Step 1: OLS regression
  reg <- lm(s1 ~ s2)
  residuals <- reg$residuals
  
  # Step 2: ADF test on residuals
  adf_result <- adf.test(residuals, alternative = "stationary")
  
  # Johansen test for robustness
  # library(urca)
  # johansen <- ca.jo(cbind(s1, s2), type = "trace", K = 2)
  
  # Hedge ratio
  hedge_ratio <- coef(reg)[2]
  
  # Half-life of mean reversion
  lag_resid <- residuals[-1]
  resid_diff <- diff(residuals)
  ar_fit <- lm(resid_diff ~ lag_resid[-length(lag_resid)])
  theta <- -coef(ar_fit)[2]
  half_life <- log(2) / theta
  
  # Spread statistics
  spread_mean <- mean(residuals)
  spread_std <- sd(residuals)
  current_zscore <- (residuals[length(residuals)] - spread_mean) / spread_std
  
  list(
    cointegrated = adf_result$p.value < 0.05,
    adf_statistic = adf_result$statistic,
    adf_pvalue = adf_result$p.value,
    hedge_ratio = hedge_ratio,
    half_life = half_life,
    spread_mean = spread_mean,
    spread_std = spread_std,
    current_zscore = current_zscore
  )
}

#' =============================================================================
#' BOOTSTRAP CONFIDENCE INTERVALS
#' =============================================================================

bootstrap_sharpe <- function(returns, n_boot = 10000, confidence = 0.95) {
  #' Bootstrap confidence intervals for Sharpe ratio
  #' Accounts for autocorrelation in returns
  
  returns <- as.numeric(returns)
  n <- length(returns)
  
  # Point estimate
  sharpe <- mean(returns) / sd(returns) * sqrt(252)
  
  # Stationary bootstrap (accounts for autocorrelation)
  library(boot)
  
  sharpe_stat <- function(data, indices) {
    r <- data[indices]
    mean(r) / sd(r) * sqrt(252)
  }
  
  boot_result <- boot(returns, sharpe_stat, R = n_boot)
  
  # BCa confidence interval
  ci <- boot.ci(boot_result, conf = confidence, type = "bca")
  
  list(
    sharpe = sharpe,
    std_error = sd(boot_result$t),
    ci_lower = ci$bca[4],
    ci_upper = ci$bca[5],
    bootstrap_distribution = boot_result$t
  )
}

#' =============================================================================
#' API ENDPOINT WRAPPER - JSON Interface
#' =============================================================================

run_analysis <- function(json_input) {
  #' Main entry point for Python/Go interop via JSON
  
  input <- fromJSON(json_input)
  
  results <- list()
  
  if (!is.null(input$returns)) {
    returns <- as.matrix(input$returns)
    
    # Run requested analyses
    if ("regime" %in% input$analyses) {
      results$regime <- detect_market_regimes(returns[,1])
    }
    
    if ("risk" %in% input$analyses) {
      results$risk <- calculate_risk_metrics(returns[,1])
    }
    
    if ("dcc_garch" %in% input$analyses && ncol(returns) >= 2) {
      results$dcc_garch <- tryCatch(
        fit_dcc_garch(returns),
        error = function(e) list(error = e$message)
      )
    }
    
    if ("factor_model" %in% input$analyses) {
      results$factor_model <- estimate_factor_model(returns)
    }
    
    if ("covariance" %in% input$analyses) {
      results$covariance <- ledoit_wolf_shrinkage(returns)
    }
  }
  
  toJSON(results, auto_unbox = TRUE, digits = 8)
}

# Print startup message
cat("Elite Quant R Statistical Engine Loaded\n")
cat("Available functions:\n")
cat("  - detect_market_regimes()\n")
cat("  - fit_dynamic_copula()\n")
cat("  - fit_dcc_garch()\n")
cat("  - ledoit_wolf_shrinkage()\n")
cat("  - estimate_factor_model()\n")
cat("  - elastic_net_alpha()\n")
cat("  - calculate_risk_metrics()\n")
cat("  - test_cointegration()\n")
cat("  - bootstrap_sharpe()\n")

