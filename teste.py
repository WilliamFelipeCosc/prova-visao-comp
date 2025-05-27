import numpy as np
from keras import layers, models, datasets
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_cats_dogs_from_cifar10():
    # Carregar o dataset CIFAR-10
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # Filtrar apenas as classes "gato" (3) e "cachorro" (5)
    train_filter = np.where((y_train == 3) | (y_train == 5))[0]
    test_filter = np.where((y_test == 3) | (y_test == 5))[0]

    x = np.concatenate([x_train[train_filter], x_test[test_filter]], axis=0)
    y = np.concatenate([y_train[train_filter], y_test[test_filter]], axis=0)

    # Re-rotular: gato=0, cachorro=1
    y = (y == 5).astype(np.float32)

    # Normalizar as imagens
    x = x.astype('float32') / 255.0

    # Split: 80% para teste, 20% para treino
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    return (x_train, y_train), (x_test, y_test)

# Modelo aprimorado com Dropout e BatchNormalization para evitar overfitting e melhorar generalização
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),  # Mais uma camada convolucional
        layers.Flatten(),
        layers.Dense(64, activation='relu'),           # Mais neurônios na densa
        layers.Dropout(0.3),                           # Dropout leve para evitar overfitting
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
(x_train, y_train), (x_test, y_test) = load_cats_dogs_from_cifar10()
class_names = ['gato', 'cachorro']

# Data augmentation mais leve:
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Treinar o modelo com mais épocas
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(x_test, y_test)
)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Acurácia no conjunto de teste: {test_acc:.4f}')

# Previsões para o conjunto de teste
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Relatório de métricas
print("\nMétricas de avaliação no conjunto de teste:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Função para classificar novas imagens
def predict_images(model, imgs_path):
    for img_path in imgs_path:
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predict_class = int(prediction[0][0] > 0.5)
        print(img_path, class_names[predict_class])

# Exemplo de uso:
predict_images(model, ['./images/shiba.jpg', './images/gato1.jpg', './images/bodercollie.jpg', './images/gato2.webp', './images/golden.jpg', './images/gato3.jpg'])