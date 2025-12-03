# A/B-Testing-Proportion-Analysis
This project performs a rigorous statistical analysis to compare proportions between two groups: a control group and an experimental group. It utilizes a Two-Sample Z-Test to determine if the observed difference in conversion rates (or similar proportions) is statistically significant.

🚀 A/B Testing: Proportion Analysis
📋 Project Overview

This project performs a statistical comparison of proportions between two groups: a control group and an experimental group.
A Two-Sample Z-Test is applied to determine whether the difference in conversion rates (or similar proportions) is statistically significant.

🧠 Statistical Methodology

The analysis includes the following steps:

Standard Error Calculation
Computing the variability of the difference between two proportions.

Z-Score Computation
Measuring how many standard errors the observed difference is from the null hypothesis.

Critical Value & P-Value Assessment
Evaluating statistical significance at a chosen confidence level (e.g., 95%).

💻 Code Breakdown

Core logic implemented using NumPy and SciPy:

Code Snippet	Description
SE = np.sqrt(pooled_variance)	Calculates the standard error.
test_stat = (...) / SE	Computes the Z-test statistic.
p_value = 2 * norm.sf(...)	Computes the two-tailed p-value.
📊 Results & Interpretation

Test Statistic: -57.36

P-Value: 0.0

Significance Level (α): 0.05

Conclusion:
Since the P-Value (0.0) < α (0.05), we reject the null hypothesis.
There is a highly statistically significant difference between the control and experimental groups.

🛠️ Prerequisites

To run the analysis, you need:

Python 3.x

Libraries:

numpy

scipy

Install dependencies:

pip install numpy scipy
