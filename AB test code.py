#%%
import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
#%%
# Load the dataset
df = pd.read_csv('ab_test_click_data.csv')
#%%
# Display first few rows
df.head()
#%%
# Summary statistics
df.describe()
#%%
# Total clicks by group
df.groupby("group")["click"].sum()
#%%
# Define color palette
palette={0:"yellow",1:"black"}
plt.figure(figsize=(10,6))
# Plot countplot of clicks by group
ax=sns.countplot(x="group", data=df, hue="click", palette=palette)
plt.title("Click Distribution in Experimental and Control groups")
plt.xlabel("Group")
plt.ylabel("Count")
plt.legend(title="Click", labels=["No", "Yes"])
#%%
# Assign a different color palette for the second plot
palette = {0: 'orange', 1: 'black'}

# Plot countplot with percentages
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='group', hue='click', data=df, palette=palette)
plt.title("Click Distribution in Control and Experimental Groups")
plt.xlabel("Group")
plt.ylabel("Click Count")
plt.legend(title="Click", labels=["No", "Yes"])

# Calculate percentages to annotate on bars
group_counts = df.groupby(["group"]).size()
group_click_counts = df.groupby(["group", "click"]).size().reset_index(name="count")

# Loop through each bar to add text
for p in ax.patches:                
    height = p.get_height()          # Get bar height (number of clicks)

    # --- Logic: Identify which group the bar belongs to ---
    # p.get_x() gives the left edge coordinate of the bar.
    # The plot is split into two groups around x=0.5.

    # If bar's x-position is < 0.5, it's the Experimental group, else Control.
    group = "exp" if p.get_x() < 0.5 else "con"

    # Bar x-position determines if it's "no click" (0) or "click" (1).
    click = 1 if p.get_x() > 0.5 else 0

    # Get total count for the identified group.
    total = group_counts[group]

    # Calculate percentage: (Bar Height / Group Total) * 100
    percentage = height / total * 100

    # --- Add Text to Bar ---
    ax.text(
        # X Position: Middle of the bar (Left edge + half width)
        p.get_x() + p.get_width() / 2,

        # Y Position: 5 units above the bar top
        height + 5,

        # Content: Calculated percentage formatted to 1 decimal place
        f'{percentage:.1f}%',

        # Alignment: Horizontal center
        ha="center",

        color="black", # Font color
        fontsize=10,   # Font size
    )
#%%
# Calculate sample sizes for each group
N_con = df[df["group"] == "con"].count()
N_exp = df[df["group"] == "exp"].count()

# Calculate total number of clicks per group by summing 1's
X_con = df.groupby("group")["click"].sum().loc["con"]
X_exp = df.groupby("group")["click"].sum().loc["exp"]

# Print calculated values for visibility
print(df.groupby("group")["click"].sum())
print("Number of users in Control: ", N_con)
print("Number of users in Experimental: ", N_exp)
print("Number of Clicks in Control: ", X_con)
print("Number of Clicks in Experimental: ", X_exp)
#%%
# Step 1: Compute estimated click probability for each group
p_con_hat = X_con / N_con
p_exp_hat = X_exp / N_exp

print("Click Probability in Control Group:", p_con_hat)
print("Click Probability in Experimental Group:", p_exp_hat)

# Step 2: Compute pooled click probability estimate
p_pooled_hat = (X_con + X_exp) / (N_con + N_exp)
print("Pooled Click Probability:", p_pooled_hat)
#%%
# --- STEP 3: Variance Calculation ---
# To perform the Z-test for statistical significance,
# we need to calculate the pooled variance.

# Apply pooled variance formula
pooled_variance = p_pooled_hat * (1 - p_pooled_hat) * (1/N_con + 1/N_exp)

print("Pooled Variance: ", pooled_variance)
#%%
# Set significance level (alpha)
alpha=0.05
# Minimum detectable effect (unused in this specific z-test snippet)
delta=0.1

# Compute standard error of the test
SE = np.sqrt(pooled_variance)
print("Standard Error is: ", SE)

# Compute Z-test statistic
Test_stat = (p_con_hat - p_exp_hat)/SE
print("Test Statistics for 2-sample Z-test is:", Test_stat)

# Determine critical value of the Z-test (two-tailed)
Z_crit = norm.ppf(1-alpha/2)
print("Z-critical value from Standard Normal distribution: ", Z_crit)
#%%
# Step 4: Calculate P-value
p_value = 2 * norm.sf(abs(Test_stat))

# Define function to check statistical significance
def is_statistical_significance(p_value, alpha):
    """
    Assesses statistical significance based on the p-value and alpha.
    
    Arguments:
    - p_value (float): The p-value resulting from a statistical test.
    - alpha (float): The significance level threshold.
    
    Returns:
    - Prints the assessment of statistical significance.
    """
    # Print p-value rounded to 3 decimal places
    print(f"P-value of the 2-sample Z-test: {round(p_value, 3)}")

    # Determine statistical significance
    if p_value < alpha:
        print("There is statistical significance, indicating that the observed differences between the groups are unlikely to have occurred by chance.")
    else:
        print("There is no statistical significance, suggesting that the observed differences between the groups could have occurred by chance.")

# Call the function to check significance
is_statistical_significance(p_value, alpha)
#%%
