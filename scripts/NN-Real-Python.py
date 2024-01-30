import numpy as np

# Wrapping the vectors in NumPy arrays
input_vector = np.array([1.66, 1.56])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])

def make_prediction(input_vector, weights, bias):
     layer_1 = np.dot(input_vector, weights) + bias
     return layer_1

prediction = make_prediction(input_vector, weights_1, bias)

print(f"The prediction result is: {prediction}")

# Changing the value of input_vector
input_vector = np.array([2, 1.5])

prediction = make_prediction(input_vector, weights_1, bias)

print(f"The prediction result is: {prediction}")


target = 1
for epoch in range(10):
     mse = np.square(prediction - target)

     derivative = 2 * (prediction - target)

     print(f"The derivative is {derivative}")

     # Updating the weights
     weights_1 = weights_1 - derivative

     prediction = make_prediction(input_vector, weights_1, bias)

     error = (prediction - target) ** 2

print(f"Prediction: {prediction}; Error: {error}")