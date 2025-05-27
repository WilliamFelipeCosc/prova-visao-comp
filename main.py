import cv2 
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split

def read_img(path, isRGB = False):
    image = cv2.imread(path, cv2.IMREAD_COLOR if isRGB else cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (128, 128))

def gaussian_blur(path):
    image = read_img(path, True)
    return cv2.GaussianBlur(image, (15, 15), 0)

def equalize(path):
    image = read_img(path)
    return cv2.equalizeHist(image)

img_shiba_blur = gaussian_blur('./images/shiba.webp')
img_shiba_hist = equalize('./images/shiba.webp')

cv2.imshow("shiba blur", img_shiba_blur)
cv2.imshow("shiba hist", img_shiba_hist)

cv2.waitKey(0)
cv2.destroyAllWindows()



# Carregando o dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalização das imagens
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Dividindo o conjunto de treinamento em 80% para treinamento e 20% para validação
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

class_names = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo',
               'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']


def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

model = build_model()

history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

def avaliate_model(model, x_val, y_val, history):
    # Avaliação
    test_loss, test_acc = model.evaluate(x_val, y_val, verbose=2)
    print(f'\nAcurácia no conjunto de teste: {test_acc:.4f}')

    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.title('Desempenho da CNN no CIFAR-10')
    plt.show()

avaliate_model(model, x_test, y_test, history)

img_path = './images/shiba.webp'  # Substitua com o caminho real

def predict_images(model, imgs_path):
    for img_path in imgs_path:
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Faz a predição
        predictions = model.predict(img_array)
        predict_class = np.argmax(predictions)
        
        print(img_path, class_names[predict_class])

predict_images(model, ['./images/shiba.webp', './images/gato1.jpg', './images/gato2.webp', './images/bodercollie.jpg', './images/golden.jpeg', './images/gato3.jpg' ])
 
    

