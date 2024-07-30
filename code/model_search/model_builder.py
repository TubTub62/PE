from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.optimizers import Adam

def model_builder(fl_n, f_size, k_size, f_act,
                  dl_n, d_size, d_act,
                  dp_b, dp_a):
    
    model = Sequential()

    # Introduce Convolution Part
    for i in range(fl_n):
        model.add(Conv1D(
            filters=f_size[i],
            kernel_size=k_size,
            activation=f_act,
            padding='same'
        ))

    if dp_b:
        model.add(Dropout(dp_a))

    # Dense Part
    model.add(Flatten())
    for i in range(dl_n):
        model.add(
            Dense(
                units=d_size[i],
                activation=d_act,
            )
        )
    model.add(Dense(2, activation='softmax'))

    #learning_rate = lr
    model.compile(
        optimizer=Adam(),
        loss="mse",
        metrics=["accuracy"]
    )
    return model