import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist#โหลดข้อมูล MNIST dataset tenserflowมีในตัว
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#printing thwe shape
print("Train_images shape:", train_images.shape)
print("Train_labels shape:", train_labels.shape)
print("Test_images shape:", test_images.shape)
print("Test_labels shape:", test_labels.shape)

#Displaying the first 10 images of dataset
fig=plt.figure(figsize=(10, 10))

nrows =3
ncols =3
for i in range(9):
    fig.add_subplot(nrows, ncols, i+1)
    plt.imshow(train_images[i])
    plt.title("Digit:{}".format(train_labels[i]))
    plt.axis(False)

plt.show()
#Convert the images to float32 and normalize to [0, 1]
train_images = train_images/ 255
test_images = test_images/ 255

print("First Label before conversion:")
print(train_labels[0])

#convert the labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

print("First Label after conversion:")
print(train_labels[0])


#Using Sequential() to build layers oneafter another
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(), #input layer
    tf.keras.layers.Dense(512, activation='relu'), #hidden layer
    tf.keras.layers.Dense(10, activation='softmax') #output layer
])

model.compile(
    loss = 'categorical_crossentropy',
    optimizer='adam',
    metrics = ['accuracy']
)

history = model.fit(
    x=train_images,
    y=train_labels,
    epochs=10
)

#showing plot for looss
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.legend(['Loss'])
plt.show()


#showing plot for accuracy
plt.plot(history.history['accuracy'],color='orange')
plt.xlabel('Epochs')
plt.legend(['Accuracy'])
plt.show()

#===part 6===
#call evalluate to find the accuracy on test images
test_loos, test_accuracy = model.evaluate(
    x=test_images,
    y=test_labels
    )

print("Test loss: %.4f"% test_loos)
print("Test accuracy: %.4f"% test_accuracy)

#===part 7===
predicted_probabilities = model.predict(test_images)
predicted_classes = tf.argmax(predicted_probabilities, axis=1).numpy()

index = 11

plt.figure(figsize=(4, 4))
plt.imshow(test_images[index], cmap='gray')
plt.title(f"True Digit: {test_labels[index]} | Predicted: {predicted_classes[index]}")
plt.axis(False)
plt.show()

print("\nProbabilities predicted for image at index", index)
print(predicted_probabilities[index])

print("Predicted class for image at index", index)
print(predicted_classes[index])