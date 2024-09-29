# Siamese Network for Face Verification

This project implements a Siamese Neural Network for facial verification using TensorFlow and OpenCV. The system captures images using a webcam, preprocesses the data, and builds a Siamese Network model to verify whether two input images are of the same person.

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Real-Time Verification](#real-time-verification)
- [License](#license)

---

## Installation

To run this project, you'll need to install the following dependencies. You can install them by running the following commands:

```bash
pip3 install tensorflow==2.17.0 opencv-python matplotlib
```
Ensure your environment also supports GPU growth handling to avoid out-of-memory (OOM) errors.
```bash
# Set GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

## Dependencies

This Project Requires the Following Dependencies

- TensorFlow (v2.17.0)
- OpenCV
- Matplotlib
- NumPy
- uuid (for generating unique filenames)

```bash
pip install -r requirements.txt
```

## Dataset

Contains 3 Folders 

- Anchor: Images of the subject you are identifying.
- Positive: Additional images of the same subject (positive matches).
- Negative: Images of other people (negative matches).

To prepare the dataset:

1. Uncompress and organize the LFW (Labeled Faces in the Wild) dataset as your negative samples.
2. Capture images using your webcam for the anchor and positive samples.

```bash
# Setup directory paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Create directories if not exist
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)
```
To extract the LFW dataset and move images to the negative folder:

```bash
!tar -xf lfw.tgz
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)
```
### Collect Positive and Anchor Samples
You can capture anchor and positive images using a webcam with OpenCV:

```bash
import uuid
import cv2

# Open webcam feed and capture images for anchor/positive
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()

    # Cut down frame to 250x250px
    frame = frame[500:1300, 500:1300, :]
    frame = cv2.flip(frame, 1)  # Flip horizontally and vertically
    
    # Capture anchor images
    if cv2.waitKey(1) & 0xFF == ord('a'):
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    
    # Capture positive images
    if cv2.waitKey(1) & 0xFF == ord('p'):
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    
    # Show image on screen
    cv2.imshow('Image Collection', frame)
    
    # Quit the capture
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
## Model

A Siamese Network is a neural network that takes two inputs and determines if they belong to the same class or not. This project builds a Siamese Network with a CNN-based architecture.

### Embedding Layer

The model architecture consists of several Conv2D layers, MaxPooling layers, and a Dense layer for the final embedding.

```bash
def make_embedding():
    inp = Input(shape=(105,105,3), name='input_image')
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    return Model(inputs=inp, outputs=d1, name='embedding')
```

### Siamese Network

The Siamese Network uses two input images, passes them through the embedding model, and computes the absolute difference between the embeddings. This is passed through a dense layer with a sigmoid activation for binary classification (same person or not).

```bash
def make_siamese_model():
    # Anchor Image
    input_image = Input(name='input_img', shape=(105,105,3))

    # Validation Image
    validation_image = Input(name='validation_img', shape=(105,105,3))

    # Siamese Distance Layer
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer([embedding(input_image), embedding(validation_image)])

    # Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
siamese_model.summary()
```
## Training

### Preprocessing

Preprocessing includes resizing, normalizing, and augmenting the images to improve model performance. You can preprocess images using the following method:

```bash
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (105,105))
    img = img / 255.0
    return img
```

### Train the Model

We use a binary cross-entropy loss function and Adam optimizer. Here’s a simplified version of the training process:

```bash
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)

    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss
```
### Training Loop 
```bash
from tensorflow.keras.metrics import Recall, Precision

def train(data, EPOCHS):
    for epoch in range(1, EPOCHS+1):
        print(f'\n Epoch {epoch}/{EPOCHS}')
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Metric objects 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx + 1)

        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)
```

### Run Training

```bash
EPOCHS = 50
train(train_data, EPOCHS)
```
## Evaluation

Once trained, you can evaluate your model by calculating recall and precision on the test data.

```bash
# Evaluate model performance
precision = Precision()
recall = Recall()

# Update precision/recall metrics based on predictions
yhat = siamese_model.predict([test_input, test_val])
precision.update_state(y_true, yhat)
recall.update_state(y_true, yhat)
```

You can visualise results :

```bash
# Visualize correct verification
plt.subplot(1,2,1)
plt.imshow(test_input[2])
plt.subplot(1,2,2)
plt.imshow(test_val[2])
plt.show()
```
## Real-Time Verification

The project includes a real-time facial verification system that captures images using a webcam and compares them with saved images to determine identity.

### Usage

To verify in real-time, press v to verify an image using the webcam feed. The model predicts whether the input image matches the stored images.

```bash
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[500:1300, 500:1300, :]
    cv2.imshow('Verification', frame)
    
    if cv2.waitKey(10
```
