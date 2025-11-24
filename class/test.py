import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# --------------------------
# 1. สร้างข้อมูลตัวอย่าง
# --------------------------
np.random.seed(42)

# คลาส 0
mean0 = [2, 2]
cov0 = [[1, 0.5], [0.5, 1]]
X0 = np.random.multivariate_normal(mean0, cov0, 100)

# คลาส 1
mean1 = [6, 5]
cov1 = [[1, -0.3], [-0.3, 1]]
X1 = np.random.multivariate_normal(mean1, cov1, 100)

# รวมข้อมูล
X = np.vstack((X0, X1))
y = np.array([0]*100 + [1]*100)

# prior probability
prior0 = 0.5
prior1 = 0.5

# --------------------------
# 2. สร้าง grid สำหรับ decision boundary
# --------------------------
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# --------------------------
# 3. คำนวณ likelihood & posterior
# --------------------------
likelihood0 = multivariate_normal.pdf(grid, mean=mean0, cov=cov0)
likelihood1 = multivariate_normal.pdf(grid, mean=mean1, cov=cov1)

posterior0 = likelihood0 * prior0
posterior1 = likelihood1 * prior1

# decision boundary: posterior0 = posterior1
Z = (posterior1 > posterior0).reshape(xx.shape)

# --------------------------
# 4. วาด decision boundary
# --------------------------
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X0[:,0], X0[:,1], c='blue', edgecolors='k', label='Class 0')
plt.scatter(X1[:,0], X1[:,1], c='red', edgecolors='k', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary using Bayes’ Rule')
plt.legend()
plt.show()
