# Multivariate-TimeSeries-Modelling
Lightweight and Data-Efficient MultivariateTime Series Forecasting using Residual-Stacked Gaussian (RS-GLinear) Architecture

## Description

we adopt the original GLinear architecture and introduce—Residual-Stacked GLinear Model (Depth =4), which is an enhanced version of the original GLinear architecture that integrates more depth in the neural network via stacked linear blocks (figure 6-b). Our proposed enhanced GLinear architecture retains the original Reversible Instance  


Time series forecasting remains a challenging yet critical research area across a wide range of domains, including finance, healthcare, meteorology, and traffic management. With the emergence of Transformer models, there has been significant interest in applying Large Language Models (LLMs) to time series forecasting, leveraging their ability to model sequential dependencies as seen in text generation. However, these models are often computationally expensive and tend to struggle in capturing essential time series characteristics such as overall trends, long-term temporal dependencies, seasonality, periodicity, noise, and stationarity. The primary objective of this research is twofold. First the project aims to develop and evaluate an enhanced version of the GLinear architecture, which we will refer to as Residual Stacked GLinear model. This new variant of the GLinear architecture is designed to improve multivariate time-series forecasting by introducing deeper neural network layers and residual connections while maintaining the core features inherent to the original GLinear framework. The second aim of our research is to explore the general applicability of RS-GLinear model to two other domains that were not considered in the GLinear baseline model introduced by Risvi et al. (2025). Most time-series implementations (Transformer-based and Linear models) we came across commonly adopt baseline implementations provided by Hugging Face repository, including our baseline GLinear model. Therefore, the RS-GLinear model developed in this study is an extended version of the codebase introduced in Risvi et al. (2025) research paper . 


![image](https://github.com/user-attachments/assets/ae8fb5f8-d264-4cea-b1b6-57d47d760559) ![image](https://github.com/user-attachments/assets/8375cd54-f892-4b5a-9984-37396f747601)



![image](https://github.com/user-attachments/assets/170cd32a-9ad6-412e-bb22-b858cca79c71)




## Acknowledgment
This code is built on the code base of LTSF-Linear Predictors (**GLinear, **NLinear**, **DLinear**). 
We would like to thank the following GitHub repository for their valuable code bases, datasets and detailed descriptions:
https://github.com/t-rizvi/GLinear.git
https://github.com/cure-lab/LTSF-Linear

This repository is and extended version and build on the official Pytorch implementation of GLinear Predictor: "Paper Link".
