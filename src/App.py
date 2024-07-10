from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np
from flask_cors import CORS
import tensorflow as tf
import os
from keras import layers, models

app = Flask(__name__)
CORS(app)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [32, 32])
    image = image / 255.0 
    return image, label

def create_dataset(data_dir, batch_size = 32):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    for (label, class_name) in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, image_name))
            labels.append(label)

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    imagelabel_ds = tf.data.Dataset.zip((path_ds, label_ds))

    imagelabel_ds = imagelabel_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    imagelabel_ds = imagelabel_ds.shuffle(buffer_size=len(image_paths))
    imagelabel_ds = imagelabel_ds.batch(batch_size)
    imagelabel_ds = imagelabel_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return imagelabel_ds, class_names

train_dataset, class_names = create_dataset('src/dataset')
test_dataset, _ = create_dataset('src/dataset2')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(class_names), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=10, validation_data=test_dataset)
model.save('image_classifier.keras')

class_names = ["among us", "s", "yippee"]

@app.route('/')
def home():
    return "Flask server is running"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    if 'file' not in request.files:
        print('No file part')
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
        return jsonify({'error': 'No selected file'})

    file_data = np.frombuffer(file.read(), np.uint8)
    img = cv.imdecode(file_data, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_resized = cv.resize(img, (32, 32))
    img_resized = img_resized / 255.0

    pred = model.predict(np.array([img_resized]))
    index = np.argmax(pred)
    prediction = class_names[index]

    print('Prediction:', prediction)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

