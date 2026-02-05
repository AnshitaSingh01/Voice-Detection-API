import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.utils import to_categorical

# Dummy MFCC-like data
# shape = (samples, height, width, channels)
X = np.random.rand(40, 20, 20, 1)

y = np.array([1, 1, 0, 0] * 10)  # 1 = AI, 0 = Human
y = to_categorical(y, num_classes=2)

model = Sequential([
    Input(shape=(20, 20, 1)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=5, verbose=1)

model.save("cnn_model.h5")
print("CNN model saved successfully")
