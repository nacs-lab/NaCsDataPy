import NaCsData
import numpy as np
from matplotlib import pyplot as plt

# Example 1: Get Survivals per site, and avg survivals per parameter with no modifications
data = NaCsData.NaCsData('20231030', '125545', 'N:\\NaCsLab\\Data\\')

load_img = np.array([2])
surv_img = np.array([4])

survs, surv_errs, unique_params = data.get_survival_prob_by_site(load_img, surv_img)

# The returned survs is nparams x nsites
nsites = survs.shape[1]
nparams = survs.shape[0]

# TODO: No support yet for ScanGroup in Python
X = unique_params

# Plot sites 1 to 5
plt.figure()
for site in range(5):
    plt.errorbar(X, survs[:,site + 1], surv_errs[:, site + 1])
plt.show()

# Now we get the average
surv_avg, surv_avg_err, unique_params = data.get_survival_prob_avg(load_img, surv_img)
plt.figure()
plt.errorbar(X, surv_avg, surv_errs)
plt.show()