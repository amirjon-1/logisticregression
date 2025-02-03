import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

x_train = np.array([[34.62365962, 78.02469282],
 [30.28671077, 43.89499752],
 [35.84740877, 72.90219803],
 [60.18259939, 86.3085521 ],
 [79.03273605, 75.34437644]])

y_train = np.array([0, 0, 0, 1, 1])

def sigmoid(z):
 g = 1/ (1+ np.exp(-1*z))
 return g


def compute_cost(x, y, w, b, *argv):

 m, n = x.shape
 totalcost = 0
 for i in range (m):
  fwbx = sigmoid(np.dot(w, x[i])+b)
  loss = (-1* y[i])*np.log(fwbx) - (1- y[i])*np.log(1-fwbx)
  totalcost += loss

 totalcost = totalcost/m
 return totalcost


def compute_gradient(X, y, w, b, *argv):
 m,n = X.shape
 dj_dw = np.zeros(w.shape)
 dj_db = 0


 for i in range (m):
  z_wb = np.dot(X[i], w)+b
  f_wb =sigmoid(z_wb)

  dj_db_i = f_wb - y[i]
  dj_db += dj_db_i
  for j in range (n):
   dj_dw_ij = (f_wb - y[i]) * X[i, j]
   dj_dw[j] += dj_dw_ij

 dj_dw = dj_dw/m
 dj_db = dj_db/m

 return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
 m = len(X)

 J_history= []
 w_history = []
 for i in range (num_iters):
  dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

  w_in = w_in - alpha * dj_dw
  b_in = b_in - alpha * dj_db

  if i<100000:
   cost = cost_function(X, y, w_in, b_in, lambda_)
   J_history.append(cost)


  if i % math.ceil(num_iters/10) == 0  or i == (num_iters-1):
   w_history.append(w_in)
   print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

  return w_in, b_in, J_history, w_history



np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

iterations = 10000
alpha = 0.001
w,b, J_history,_ = gradient_descent(x_train ,y_train, initial_w, initial_b,
                                   compute_cost, compute_gradient, alpha, iterations, 0)

# plot_decision_boundary(w, b, x_train, y_train)
# # Set the y-axis label
# plt.ylabel('Exam 2 score')
# # Set the x-axis label
# plt.xlabel('Exam 1 score')
# plt.legend(loc="upper right")
# plt.show()


def predict(X, w, b):

 m, n = X.shape
 p = np.zeros(m)


 for i in range(m):
  z_wb = np.dot(X[i], w) + b
  f_wb = sigmoid(z_wb)

  if (f_wb >= 0.5):
   p[i] = 1
  else:
   p[i] =0

 return p

# regularization
def compute_cost_reg(X, y, w, b, lambda_=1):

 m, n = X.shape

 cost_without_reg = compute_cost(X, y, w, b)

 reg_cost = 0.


 for i in range(n):
  reg_cost += w[i] ** 2

 reg_cost *= lambda_
 reg_cost /= 2 * m;

 total_cost = cost_without_reg + reg_cost

 return total_cost
