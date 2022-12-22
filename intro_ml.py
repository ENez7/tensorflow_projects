from tensorflow import keras
import numpy as np

# Define the model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Compile the model
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
# Explain model.compile in the next 5 lines
# optimizer = 'sgd' means that the model will use the 
# Stochastic Gradient Descent algorithm to train the model

# loss = 'mean_squared_error' means that the model will use the mean squared error 
# algorithm to calculate the loss
# The loss is the difference between the predicted value and the actual value
# The loss is used to train the model
# The lower the loss, the more accurate the model will be
# The loss is calculated for each epoch
# The loss is the average of the losses for each epoch


# Define the data input
xs = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype = float)
# Define the correct data output
ys = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0], dtype = float)

# Train the model
# The more epochs, the more accurate the model will be
# But it will take longer to train
# The loss should be as low as possible
# The loss is the difference between the predicted value and the actual value
model.fit(xs, ys, epochs = 1000)

# Predict the output for a new input
print(f'y = 2(10) - 1 = {model.predict([10.0])}')