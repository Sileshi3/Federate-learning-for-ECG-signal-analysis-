


class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
    
def oneD_CNN_model(input_shape,num_classes):
    # Define the model
        model = keras.models.Sequential()
        # Add a 1D convolutional layer with 32 filters, kernel size 3, and ReLU activation
        model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_shape, 1)))
        # You need to specify the input_shape which is (sequence_length, input_dim), where input_dim is 1 for a single channel.

        # Add a max-pooling layer with pool size 2
        model.add(MaxPooling1D(pool_size=2))

        # Flatten the output of the convolutional layer
        model.add(Flatten())

        # Add one or more fully connected (dense) layers
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # The last layer should have as many units as the number of classes in your classification problem.

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Print a summary of the model's architecture
        model.summary()
        return model 