
# Predicting standardized absolute returns using rolling-sample textual modelling

[![DOI](https://zenodo.org/badge/425697235.svg)](https://zenodo.org/badge/latestdoi/425697235)

# Abstract

Understanding how textual information impacts financial market volatility has been one of the growing topics in financial econometric research. In this paper, we aim to examine the relationship between the volatility measure that is extracted from GARCH modeling and textual news information that is both publicly available and from subscription and the performances of the two datasets are also brought into comparison. We utilize latent Dirichlet allocation method to capture the dynamic features of the textual data overtime by summarizing their statistical outputs, such as topic distributions in documents and word distributions in topics. In addition, we transform various measures representing the popularity and diversity of topics to form predictors for rolling regression model to assess the usefulness of textual information. The proposed method captures the statistical properties of textual information from different time periods and its performance is evaluated in an out-of-sample analysis. 

Our results show that the topic measures can be more useful for predicting our volatility proxy, the unexplained variance from GARCH model than the simple moving average. The finding indicates that our method can be helpful in extracting significant textual information to improve the prediction of stock market volatility. 

<hr />

# Program details

## DataCleansing-public.py
- The data cleansing process of the public data.

## DataCleansing-private.py
- The data cleansing process of the private data.

## RiskModelling.py
- Shows how we transfer the HSI data to risk using GARCH modelling.

## rolling_LDA.py
- Contains a class of the rolling LDA process as stated on the paper.
- The formation of the customised topic scores are also embedded in this class.

## rolling_analysis.py
- Contains a class to do some preliminary analysis on the rolling window data.
- Most of the functions are not made into use for the outcome of the result discussed on the paper.

## regression.py
- Contains a class that utilise the result from rolling LDA to perform a regression-based prediction on the risk which is drawn from RiskModelling.py
- Also include the generation of report to tell which threshold should be optimal to use in that particular dataset.

## result_analysis.py
- Analyze the result produced in regression.py with the help of graphs like histogram, heat map, confusion matrix, and word cloud.
- The related analysis are discussed in the paper to show how our proposed method is helpful on the prediction of market volatility.

## sample_run_demo.ipynb
- A full demo to show how we use the rolling_LDA.py, rolling_analysis.py, regression.py, and result_analysis.py to perform prediction and related analysis.
- This version is our original version to perform the analysis. The codes will be duplicated with rolling_LDA.py, rolling_analysis.py, regression.py, and result_analysis.py. Those .py files are trying to make each step clearer. Since our proposed method involves quite a number of visualisation, so we will encourage running jupyter notebook in order to get a better understanding of the entire flow.

<hr />
