#!/usr/bin/env python
# coding: utf-8

# In[1]:


x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,]
y = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12]
mean_x=sum(x)/len(x)
mean_y=sum(y)/len(y)
a=0
sum_square_x=0
sum_square_y=0
for i in range(len(x)):
    a+=(x[i]-mean_x)*(y[i]-mean_y)
    sum_square_x+=((x[i]-mean_x)**2)
    sum_square_y+=((y[i]-mean_y)**2)
beta_1=a/sum_square_x
beta_0=mean_y-(beta_1*mean_x)
y_slope=[]
for i in range(len(x)):
    y_slope.append(beta_0+(beta_1*x[i]))
sum_square_errors=0
for i in range(len(y)):
    sum_square_errors+=((y[i]-y_slope[i])**2)
r_square=1-(sum_square_errors/sum_square_y)
print("Beta_0 :",beta_0)
print("Beta_1 :",beta_1)
print("Sum_Square_Errors :",sum_square_errors)
print("R_Square :",r_square)


# In[8]:


import numpy as np

def gradient_descent(x, y, lr, epochs, batch_size=None):
    n = len(x)
    b0 = 0
    b1 = 0
    losses = []

    for epoch in range(epochs):
        y_pred = b0 + b1 * x

        grad_b0 = -2 * np.sum(y - y_pred) / n
        grad_b1 = -2 * np.sum(x * (y - y_pred)) / n

        b0 -= lr * grad_b0
        b1 -= lr * grad_b1

        loss = np.sum((y - y_pred) ** 2) / n
        losses.append(loss)

        if batch_size is not None and (epoch + 1) % batch_size == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    return b0, b1, losses

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

lr = 0.01
epochs = 1000

b0_full, b1_full, losses_full = gradient_descent(x, y, lr, epochs)

b0_stochastic, b1_stochastic, losses_stochastic = gradient_descent(x, y, lr, epochs, batch_size=1)

y_pred_full = b0_full + b1_full * x
sse_full = np.sum((y - y_pred_full) ** 2)
r_squared_full = 1 - (sse_full / np.sum((y - np.mean(y)) ** 2))

y_pred_stochastic = b0_stochastic + b1_stochastic * x
sse_stochastic = np.sum((y - y_pred_stochastic) ** 2)
r_squared_stochastic = 1 - (sse_stochastic / np.sum((y - np.mean(y)) ** 2))

print("Full Dataset:")
print("Beta_0:", b0_full, "Beta_1:", b1_full)
print("SSE:", sse_full)
print("R^2:", r_squared_full)


# In[ ]:




