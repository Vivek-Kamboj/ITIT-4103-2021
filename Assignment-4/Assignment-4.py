
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv("https://raw.githubusercontent.com/AlfTang/Linear-Regression/master/ex1data1.txt",names = ['population','profit'],header = None)
data.head(5)
data.size
X = np.array(data.iloc[:,0]) 
y = np.array(data.iloc[:,1]) 

m = len(y)
print(m)
plt.scatter(X,y)
plt.title('Population vs Profit')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

ones = np.ones((m,1))

X = np.stack([np.ones(m), X], axis=1)

def computeCost(X,y,theta):
    J = (np.sum(np.power((np.dot(X, theta) - y),2)))/(2*m)
    return J

theta = np.zeros(2)
J = computeCost(X, y, theta)
print('With theta = [0, 0] \nCost computed = %.2f' % J)
theta = np.array([-1,2])
J = computeCost(X, y, theta)
print('With theta = [-1, 2]\nCost computed = %.2f' % J)

def gradientDescent(X, y, theta, alpha, iterations):
    m = y.size
    J_history = np.zeros(iterations)

    for i in np.arange(iterations):
        h = X.dot(theta)
        theta = theta - alpha * (1 / m) * (X.T.dot(h - y))
        J_history[i] = computeCost(X, y, theta)
    return (theta, J_history)
print('\nRunning Gradient Descent ...\n')

iterations = 2000
alpha = 0.01

theta, Cost_J = gradientDescent(X, y, theta, alpha, iterations)
print('theta:', theta.ravel())

plt.plot(Cost_J)
plt.xlabel('Iterations')
plt.ylabel('Cost_J')
plt.show()
plt.scatter(X[:,1],y, label = 'Training Data')
plt.title('Population vs Profit')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], X.dot(theta),color='red',label='Hypothesis: h(x) = %0.2f + %0.2fx'%(theta[0],theta[1]))
plt.legend(loc='best')
plt.show()

predict1 = np.dot([1, 3.5],theta) # takes inner product to get y
predict2 = np.dot([1, 7],theta) # takes inner product to get y

print('For population = 35,000, we predict a profit of ', predict1*10000)
print('For population = 70,000, we predict a profit of ', predict2*10000)

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)


J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computeCost(X, y, [theta0, theta1])

J_vals = J_vals.T

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Surface')

ax = plt.subplot(122)
plt.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.scatter(theta[0], theta[1])
plt.title('Contour, showing minimum')
print(theta)
plt.show()