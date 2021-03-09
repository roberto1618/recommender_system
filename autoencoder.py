# Model to calculate recommendations. The autoencoder expects binary vectors which the purchases
def autoencoder_model(x_train, x_test, epochs = 50, batch_size = 64, encoding_dim = 25, hidden_activation = 'relu', optimizer = 'adam', loss = 'binary_crossentropy'):
    input_dim = x_train.shape[1]
    input_layer = Input(shape = (input_dim,))
    encoding_layer = Dense(encoding_dim, activation = hidden_activation)
    encoded = encoding_layer(input_layer)

    decoding_layer = Dense(x_train.shape[1], activation = 'softmax')
    decoded = decoding_layer(encoded)

    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer = optimizer, loss = loss)

    autoencoder.fit(x_train, x_train, epochs = epochs, batch_size = batch_size, validation_data = (x_test, x_test), verbose = 0)