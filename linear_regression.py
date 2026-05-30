import numpy as np

kids = np.array([15, 27, 30, 55])
bags = np.array([5, 8, 9, 19])

m = 0.0
b = 0.0
learning_rate = 0.0001

for epoch in range(200):
    predicted_bags = m * kids + b
    error = predicted_bags - bags
    mse = (error ** 2).mean()

    gradient_m = (2 * error * kids).mean()
    gradient_b = (2 * error).mean()

    m = m - learning_rate * gradient_m
    b = b - learning_rate * gradient_b

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {mse:.4f} | gradient_m: {gradient_m:.4f}")

predicted_bags = m * kids + b
error = predicted_bags - bags
squared_errors = error ** 2

mse = squared_errors.mean()

print("squared error:", squared_errors)
print("loss (MSE):", mse)
print("actual:  ", bags)
print("error:", error)
print("predicted:", predicted_bags)
