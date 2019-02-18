import numpy as np
import matplotlib.pyplot as plt

train_X = np.array([[0,0],[0,1],[1,0],[1,1]]).T
train_Y = np.array([[0,1,1,0]])
test_X = np.array([[0,0],[0,1],[1,0],[1,1]]).T
test_Y = np.array([[0,1,1,0]])

learning_rate = 0.1
S = 5

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

S0, S1, S2 = 2, 5, 1
m = 4

w1 = np.random.randn(S1, S0)
b1 = np.zeros((S1, 1))
w2 = np.random.randn(S2, S1)
b2 = np.zeros((S2, 1))
print("b1=", b1)
print("w2=", w2)

print("w1=", w1)
print("b2=", b2)

Js = list()
for i in range(6000):
    Z1 = np.dot(w1, train_X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(Z2)

    J = np.sum(-train_Y * np.log(A2) + (train_Y-1) * np.log(1-A2)) / m
    Js.append(J)
    dZ2 = A2 - train_Y
    dW2 = np.dot(dZ2, A1.T) / m
    dB2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = np.dot(w2.T, dZ2) * sigmoid_derivative(Z1)
    dW1 = np.dot(dZ1, train_X.T) / m
    dB1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    learning_rate = 0.1
    w1 = w1 - dW1 * learning_rate
    w2 = w2 - dW2 * learning_rate
    b1 = b1 - dB1 * learning_rate
    b2 = b2 - dB2 * learning_rate


plt.plot(Js)
plt.show()