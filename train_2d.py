# !pip install tensorflow

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Taille d'entrée des données
input_shape = (240, 240, 1) # Image 2D

def Unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    up1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c3)
    cc2 = layers.concatenate([up1, c2], axis=3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(cc2)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    up2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c4)
    cc3 = layers.concatenate([up2, c1], axis=3)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cc3)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    # Sortie
    outputs = layers.Conv2D(5, (1, 1), activation='softmax')(c5)  # 4 classes

    model = models.Model(inputs, outputs)
    return model

# Charger les images et les masques
def load_image_mask(image_path, mask_path):
    """
    Charge une image et son masque (fichier segmentation) à partir de fichiers .npy
    """
    img = np.load(image_path)
    mask = np.load(mask_path)

    # Normalisation des images (0-1)
    i_min, i_max = np.min(img), np.max(img)
    img = (img - i_min) / (i_max - i_min) if i_max != i_min else img

    return img, mask

global_path = "/kaggle/input/2d-brain/"
os.chdir(global_path)

# Dossiers des images 2D (slices) et des masques de segmentation
image_dir = "slices_2D_Flair"
mask_dir = "slices_2D_Seg"

# Listes des fichiers d'images et de masques (fichiers .npy) pour s'assurer que les deux sont chargés dans le même ordre
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

images = []
masks = []

# Charger toutes les images et leurs masques
for image_file, mask_file in zip(image_files, mask_files):
    img, mask = load_image_mask(os.path.join(image_dir, image_file), os.path.join(mask_dir, mask_file))
    images.append(img)
    masks.append(mask)

images = np.array(images)
masks = np.array(masks)

# Assurer la forme des données : (batch_size, height, width, channels)
images = np.expand_dims(images, axis=-1)  # Ajouter une dimension pour les canaux (1 canal pour chaque image et masque)
masks = np.expand_dims(masks, axis=-1)

# Séparer les données en ensembles de training et de validation
X_train, X_val, Y_train_0, Y_val_0 = train_test_split(images, masks, test_size=0.175, random_state=1)

# Convertir les labels en one-hot
Y_train = to_categorical(Y_train_0, num_classes=5)
Y_val = to_categorical(Y_val_0, num_classes=5)

import tensorflow as tf
print("GPU dispo :", tf.config.list_physical_devices('GPU'))

# Création du modèle
model = Unet(input_shape)

# Afficher un résumé du modèle
# model.summary()

# Compilation du modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Les masques contiennent les valeurs [0,1,2,4]
              metrics=['accuracy'])

# Callbacks pour surveiller l'entraînement
checkpoint = ModelCheckpoint('/kaggle/working/unet_2D_best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Commencer à mesurer le temps
start_time = time.time()

# Entraîner le modèle
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, Y_val),
    callbacks= [checkpoint, early_stopping],  # Callbacks pour sauvegarde et arrêt anticipé
    verbose=1 # 0 : rien, 1 : barre de progression 2 : n° de l'epoch
)

# Calculer le temps écoulé (temps de l'entraînement)
end_time = time.time()
train_time = end_time - start_time
print(f"Temps d'entraînement : {train_time:.2f} secondes")

# Récupérer les valeurs de l'historique
history_dict = history.history  # history est l'objet retourné par model.fit()

# Affichage de la perte (loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_dict['loss'], label='Train Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.title('Courbe de perte (Loss)')
plt.legend()

# Affichage de la précision (si applicable)
if 'accuracy' in history_dict:  # Vérifie si accuracy est suivi
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Train Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.title('Courbe de précision (Accuracy)')
    plt.legend()

plt.show()

# Sauvegarder le modèle
# model.save('unet_2D_best.keras')

import os
from IPython.display import FileLink

# Téléchargez le modèle
# os.chdir('/kaggle/working/')
# FileLink(r'unet_2D_best_model.keras')