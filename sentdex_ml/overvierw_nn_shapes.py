
1. ############################################################################################################################

abalone_model = tf.keras.Sequential([
  layers.Dense(64,input_shape = (7,)),
  # layers.Dense(64), #no input shape works too
  layers.Dense(1)
])
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 normalization (Normalizatio  (None, 7)                15
 n)

 dense_6 (Dense)             (None, 64)                512

 dense_7 (Dense)             (None, 1)                 65

=================================================================
Total params: 592
Trainable params: 577
Non-trainable params: 15
_________________________________________________________________


abalone_features
array([[0.435, 0.335, 0.11 , ..., 0.136, 0.077, 0.097],
       [0.585, 0.45 , 0.125, ..., 0.354, 0.207, 0.225],
       [0.655, 0.51 , 0.16 , ..., 0.396, 0.282, 0.37 ],
       ...,
       [0.53 , 0.42 , 0.13 , ..., 0.374, 0.167, 0.249],
       [0.395, 0.315, 0.105, ..., 0.118, 0.091, 0.119],
       [0.45 , 0.355, 0.12 , ..., 0.115, 0.067, 0.16 ]])
type(abalone_features)
#<class 'numpy.ndarray'>
(3320, 7)




2. ############################################################################################################################

    def build_model1_two_hidden_layers():
        model = Sequential()
        model.add(Flatten(input_shape = (normed_train_data.shape[1],))),
        model.add(Dense(16))     
        # model.add(Dense(16, input_shape = (normed_train_data.shape[1],)))    # Input layer => input_shape must be explicitly designated       
        model.add(Dense(16,Activation('relu'))) # Hidden layer 1 => only output dimension should be designated (output dimension = # of Neurons = 50)
        model.add(Dense(3, activation='softmax'))                          # Output layer => output dimension = 1 since it is a regression problem
        # output neurons must be == number of possible classes
        # Activation: sigmoid, softmax, tanh, relu, LeakyReLU. 
        learning_rate = 0.0001
        optimizer = optimizers.Adam(learning_rate)
        # was macht compile()????????
        model.compile(loss='categorical_crossentropy',#from_logits=True),
                    optimizer=optimizer,
                    metrics=['accuracy']) # for regression problems, mean squared error (MSE) is often employed
        return model

model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 4)                 0

 dense (Dense)               (None, 16)                80

 dense_1 (Dense)             (None, 16)                272

 dense_2 (Dense)             (None, 3)                 51

=================================================================
Total params: 403
Trainable params: 403
Non-trainable params: 0
_________________________________________________________________

normed_train_data.head
<bound method NDFrame.head of      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
30           -1.231870          0.111684          -1.182821         -1.260737
25           -0.990327         -0.108022          -1.182821         -1.260737
89           -0.386469         -1.206549           0.173280          0.180105
93           -0.990327         -1.645960          -0.222250         -0.212852
34           -1.111098          0.111684          -1.239325         -1.391723
..                 ...               ...                ...               ...
108           1.062790         -1.206549           1.190355          0.835034
29           -1.352641          0.331389          -1.182821         -1.260737
128           0.700475         -0.547433           1.077347          1.227991
9            -1.111098          0.111684          -1.239325         -1.391723
149           0.096617         -0.108022           0.794826          0.835034

[120 rows x 4 columns]>
print(normed_train_data.shape)
(120, 4)
print((normed_train_data.shape[1],))
(4,)


3. ############################################################################################################################

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), #input shape is 28x28
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 784)               0

 dense (Dense)               (None, 128)               100480

 dropout (Dropout)           (None, 128)               0

 dense_1 (Dense)             (None, 10)                1290

=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________


array([[[0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        ...,
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.]]])
shape
(60000, 28, 28)
len
47040000


4. ############################################################################################################################
#C:\Users\reifv\root\Heidelberg Master\vs_codes\tensorflow_xla\tensorflow\compiler\xla\g3doc\tutorials\autoclustering_xla.ipynb

#V: ##################################################
import tensorflow as tf
import tensorflow_datasets as tfds

def load_data():
  result = tfds.load('cifar10', batch_size = -1)
  (x_train, y_train) = result['train']['image'],result['train']['label']
  (x_test, y_test) = result['test']['image'],result['test']['label']
  
  x_train = x_train.numpy().astype('float32') / 256
  x_test = x_test.numpy().astype('float32') / 256

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
  return ((x_train, y_train), (x_test, y_test))

(x_train, y_train), (x_test, y_test) = load_data()

def generate_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Activation('softmax')
  ])

model = generate_model()
print(model.summary())

def compile_model(model):
  opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
  return model

model = compile_model(model)

# def train_model(model, x_train, y_train, x_test, y_test, epochs=2):

# def warmup(model, x_train, y_train, x_test, y_test):
#   # Warm up the JIT, we do not wish to measure the compilation time.
#   initial_weights = model.get_weights()
#   train_model(model, x_train, y_train, x_test, y_test, epochs=1)
#   model.set_weights(initial_weights)

# warmup(model, x_train, y_train, x_test, y_test)
# %time 
# train_model(model, x_train, y_train, x_test, y_test)
epochs = 2
model.fit(x_train, y_train, batch_size=256, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



model.summary()
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_4 (Conv2D)           (None, 32, 32, 32)        896

 activation_6 (Activation)   (None, 32, 32, 32)        0

 conv2d_5 (Conv2D)           (None, 30, 30, 32)        9248

 activation_7 (Activation)   (None, 30, 30, 32)        0

 max_pooling2d_2 (MaxPooling  (None, 15, 15, 32)       0
 2D)

 dropout_3 (Dropout)         (None, 15, 15, 32)        0

 conv2d_6 (Conv2D)           (None, 15, 15, 64)        18496

 activation_8 (Activation)   (None, 15, 15, 64)        0

 conv2d_7 (Conv2D)           (None, 13, 13, 64)        36928

 activation_9 (Activation)   (None, 13, 13, 64)        0

 max_pooling2d_3 (MaxPooling  (None, 6, 6, 64)         0
 2D)

 dropout_4 (Dropout)         (None, 6, 6, 64)          0

 flatten_1 (Flatten)         (None, 2304)              0

 dense_2 (Dense)             (None, 512)               1180160

 activation_10 (Activation)  (None, 512)               0

 dropout_5 (Dropout)         (None, 512)               0

 dense_3 (Dense)             (None, 10)                5130

 activation_11 (Activation)  (None, 10)                0

=================================================================
Total params: 1,250,858
Trainable params: 1,250,858
Non-trainable params: 0
_________________________________________________________________

x_train
array([[[[0.55859375, 0.375     , 0.2734375 ],
         [0.55078125, 0.375     , 0.28125   ],
         [0.52734375, 0.36328125, 0.28125   ],

         [0.49609375, 0.58984375, 0.484375  ],
         [0.5234375 , 0.625     , 0.546875  ],
         [0.4609375 , 0.578125  , 0.49609375]]]], dtype=float32)
x_train.shape
(50000, 32, 32, 3)
y_train.shape
(50000, 10)
y: 10 classes, aber binary matrix form






@for search every symbol
or ctr shift .

to dos:
DONE 1. titanic example selber in einem notebook!!! kleiner nachmachen, oder anderen datensatz:)
DONE 2. string cols als input geht, aber so wiie auch oft, ist accuracy schlecht und wird nicht besser
-->  3. tf adv tuts beide machen 
4. ein nn laufen lassen wo daten von tf.data kommen!!! iris = tf.convert_to_tensor(pd.read_csv("iris.csv"))
5. alle tabs im 1. chrome window durchsehen: input shapes mehr lernen
6. rausfinden warum tw so schlechte fits bei data loader bsp
7. hyperparameter tuning? --> tensorboard
8. möglichkeit logs summary zu bekommen was von 30+ modellen das beste ist?

python programming.net tut stur durchballern?
iris it obejct col trainieren!!!!!!!!! weil scheiss acc davor. und evtl aus csv.ipnyb oder so noch andere 
schlechte modelle suchen


die optimizer mal probieren

SGD
RMSprop
Adam
Adadelta
Adagrad
Adamax
Nadam
Ftrl