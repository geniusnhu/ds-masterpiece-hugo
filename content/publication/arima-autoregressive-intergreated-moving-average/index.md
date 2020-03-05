---
authors:
- admin
categories:
- Data Science
- Forecasting
- ARIMA
date: "2020-01-26T00:00:00Z"
draft: false
featured: false
image:
  caption: ''
  focal_point: ""
  placement: 2
  preview_only: false
lastmod: "2019-11-26T00:00:00Z"
projects: []
subtitle: 'ARIMA Autoregressive Integrated Moving Average model'
summary: 'Comprehensive summary of ARIMA model and how to apply it to forecasting'
tags:
- Data Science
- Forecasting
- ARIMA
title: 'ARIMA Autoregressive Integrated Moving Average model'
---

## 1.	Concept Introduction

Auto Regressive Integrated Moving Average: 'explains' a given time series based on its own past values. ARIMA is expressed as $ARIMA(p,d,q)$

There are 3 parts in the ARIMA model: **Auto Regressive (AR)** $p$, **Integrated (I)** $d$, **Moving Average (MA)** $q$

*	**Integrated** (Or stationary): how is your data depend on each other across time. One of the characteristics of Stationary is that the effect of an observation dissipated as time goes on. Therefore, the best long-term predictions for data that has stationary is the historical mean of the series.

...To get stationary to your data, we need Differencing (or the change from one time period to another)  
*	**Auto Regressive**: deals with previous values of model, or called lags, and there are unlimited number of lags in the model. The basic assumption of this model is that the current series value depends on its previous values. This is the long memory model because the effect slowly dissipates across time. p is preferred as the maximum lag of the data series.
... The AR can be denoted as
$Y_{t}=\omega_{0}+\alpha_{1}Y_{t-1}+\alpha_{2}Y_{t-2}+...+\xi$

*	**Moving Average**: deal with 'shock' or error in the model, or how abnormal your current value is compared to the previous values (has some residual effect). This is short memory model because the effect quickly disappears completely.

p, d, and q are non-negative integers; 
*	$p$: the order (number of time lags) of the autoregressive model, also called the lag order.
*	$d$: the degree of differencing (the number of times the data have had past values subtracted)
*	$q$: the order of the moving-average model (The size of the moving average window)

A value of 0 can be used for a parameter, which indicates to not use that element of the model. When two out of the three parameters are zeros, the model may be referred to non-zero parameter. For example, $ARIMA (1,0,0)$ is $AR(1)$  (i.e. the ARIMA model is configured to perform the function if a AR model), $ARIMA(0,1,0)$ is $I(1)$, and $ARIMA(0,0,1)$ is $MA(1)$

## 2. Model evaluation

There are 2 common measures to evaluate the predicted values with the validation set.

**1.	Mean Absolute Error (MAE):**
...How far your predicted term to the real value on absolute term. One of the drawbacks of the MAE is because it shows the absolute value so there is no strong evidence and comparison on which the predicted value is actually lower or higher. 

$MAE=\frac{1}{n}\sum_{i = 1}^{n} |Y_{t}-\hat{Y_{t}}|$

can be run with R
```r
mean(abs(Yp - Yv))
```

or in Python
```python
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred)
```

**2. Mean absolute percentage error (MAPE):**

... The MAE score shows the absolute value and it is hardly to define whether that number is good or bad, close or far from expectation. This is when MAPE comes in.
...MAPE measures how far your predicted term to the real value on absolute percentage term.

$MAPE=100\frac{1}{n}\sum_{i = 1}^{n} \frac{|Y_t-\hat{Y_t}|} {\hat{Y_{t}}}$

Can compute as
```
100 x mean(abs(Yp - Yv) / Yv )
```
