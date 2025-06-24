import numpy as np
from scipy.stats import wilcoxon


#DoS0 results
DoS0 = [54.67, 56.65, 44.46, 64.13, 70.30, 69.93 ]
print(np.mean(DoS0))#Mean of all 3 DoS runs
DoS2_avg = [81.56,83.86,68.57,70.73,89.07,69.86]
DoS4 = [78.83,67.51,36.09,59.87,81.45,69.96]
hi =[DoS0[count]-DoS4[count] for count in range(len(DoS2_avg))]
print(hi)

# Perform wilcoxon test
statistic, p_value = wilcoxon(DoS0, DoS4)
print(f"W: {statistic:.4f}")
print(f"p-value: {p_value:.6f}")
print(p_value/2, " one sided")
