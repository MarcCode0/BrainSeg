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
# input_shape = (240, 240, 155, 1)  # Image 3D de taille (240, 240, 155)
# input_shape = (240, 240, 1) # Image 2D
input_shape = (240, 240, 4)  # Images 2D avec 4 canaux : T1, T2, T1ce, Flair

# Bloc d'attention pour U-Net
def attention_block(x, g, inter_channels):
    """
    x = Entrée de l'encodeur
    g = Entrée du décodeur
    inter_channels = Nombre de filtres intermédiaires
    """
    theta_x = layers.Conv2D(inter_channels, (1, 1), padding='same')(x)  # Réduction de dimension
    phi_g = layers.Conv2D(inter_channels, (1, 1), padding='same')(g)  # Mise à l'échelle du skip connection
    add = layers.Add()([theta_x, phi_g])  # Fusion
    activation = layers.Activation('relu')(add)  # Non-linéarité
    psi = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(activation)  # Génération du masque d'attention
    return layers.Multiply()([x, psi])  # Application du masque

def Unet(input_shape, num_classes=5):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    # c1 = layers.BatchNormalization()(c1)  # Normalisation
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    # c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    # c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    # c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    # c3 = layers.Dropout(0.25)(c3)  # Ajout du dropout (30%), 25% à tester

    # Decoder
    up1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c3)
    att1 = attention_block(c2, up1, 128)  # Appliquer l'attention sur la connexion skip
    cc2 = layers.concatenate([up1, att1], axis=3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(cc2)
    # c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    # c4 = layers.BatchNormalization()(c4)

    up2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c4)
    att2 = attention_block(c1, up2, 64)
    cc3 = layers.concatenate([up2, att2], axis=3)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cc3)
    # c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    # c5 = layers.BatchNormalization()(c5)

    # Sortie
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)  # 5 classes

    model = models.Model(inputs, outputs)
    return model

def conv_block(x, filters, kernel_size=(3, 3), activation='relu', bn=False):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if bn:
        x = layers.BatchNormalization()(x)  # Normalisation
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if bn:
        x = layers.BatchNormalization()(x)  # Normalisation
    x = layers.Activation(activation)(x)

    return x

# U-Net++
def Unet_plus_plus(input_shape, num_classes=5):
    inputs = layers.Input(input_shape)

    # Encoder
    c_00 = conv_block(inputs, 64)
    p0 = layers.MaxPooling2D((2, 2))(c_00)

    c_10 = conv_block(p0, 128)
    p1 = layers.MaxPooling2D((2, 2))(c_10)

    c_20 = conv_block(p1, 256)
    p2 = layers.MaxPooling2D((2, 2))(c_20)

    c_30 = conv_block(p2, 512)
    p3 = layers.MaxPooling2D((2, 2))(c_30)

    # Bottleneck
    c_40 = conv_block(p3, 1024)
    c_40 = layers.Dropout(0.3)(c_40)

    # Decoder avec denses skip connections (U-Net++)
    merge_01 = layers.Concatenate(axis=-1)([c_00, layers.UpSampling2D((2, 2))(c_10)])
    c_01 = conv_block(merge_01, 64)
    merge_11 = layers.Concatenate(axis=-1)([c_10, layers.UpSampling2D((2, 2))(c_20)])
    c_11 = conv_block(merge_11, 128)
    merge_21 = layers.Concatenate(axis=-1)([c_20, layers.UpSampling2D((2, 2))(c_30)])
    c_21 = conv_block(merge_21, 256)
    merge_31 = layers.Concatenate(axis=-1)([c_30, layers.UpSampling2D((2, 2))(c_40)])
    c_31 = conv_block(merge_31, 512)
    # c_31 = layers.Dropout(0.2)(c_31)

    merge_02 = layers.Concatenate(axis=-1)([c_00, c_01, layers.UpSampling2D((2, 2))(c_11)])
    c_02 = conv_block(merge_02, 64)
    merge_12 = layers.Concatenate(axis=-1)([c_10, c_11, layers.UpSampling2D((2, 2))(c_21)])
    c_12 = conv_block(merge_12, 128)
    merge_22 = layers.Concatenate(axis=-1)([c_20, c_21, layers.UpSampling2D((2, 2))(c_31)])
    c_22 = conv_block(merge_22, 256)
    # c_22 = layers.Dropout(0.2)(c_22)

    merge_03 = layers.Concatenate(axis=-1)([c_00, c_01, c_02, layers.UpSampling2D((2, 2))(c_12)])
    c_03 = conv_block(merge_03, 64)
    merge_13 = layers.Concatenate(axis=-1)([c_10, c_11, c_12, layers.UpSampling2D((2, 2))(c_22)])
    c_13 = conv_block(merge_13, 128)
    # c_13 = layers.Dropout(0.2)(c_13)

    merge_04 = layers.Concatenate(axis=-1)([c_00, c_01, c_02, c_03, layers.UpSampling2D((2, 2))(c_13)])
    c_04 = conv_block(merge_04, 64)
    # c_04 = layers.Dropout(0.2)(c_04)

    # Sortie
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c_04)

    model = models.Model(inputs, outputs)
    return model

# Charger les images et les masques (une seule modalité)
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

# Charger les images et les masques pour plusieurs modalités
def load_multimodal_images(t1_path, t2_path, t1ce_path, flair_path, mask_path):
    """
    Charge 4 modalités d'IRM et un masque à partir de fichiers .npy.
    """
    t1 = np.load(t1_path)
    t2 = np.load(t2_path)
    t1ce = np.load(t1ce_path)
    flair = np.load(flair_path)
    mask = np.load(mask_path)

    # Normalisation individuelle de chaque modalité (0-1)
    def normalize(img):
        i_min, i_max = np.min(img), np.max(img)
        return (img - i_min) / (i_max - i_min) if i_max != i_min else img

    t1, t2, t1ce, flair = map(normalize, [t1, t2, t1ce, flair])

    # Empiler les modalités pour obtenir un tensor de forme (240, 240, 4)
    img_multi = np.stack([t1, t2, t1ce, flair], axis=-1)

    return img_multi, mask

#---------------------------------------------------------------------------------------#

# Se placer dans le répertoire de travail
global_path = "/kaggle/input/2d-brain/"
os.chdir(global_path)

# Dossiers des images 2D (slices) et des masques de segmentation
# image_dir = "slices_2D_Flair"
image_dirs = {
    "T1": "slices_2D_T1",
    "T2": "slices_2D_T2",
    "T1ce": "slices_2D_T1ce",
    "Flair": "slices_2D_Flair"
}
mask_dir = "slices_2D_Seg"

# Listes des fichiers d'images et de masques (fichiers .npy) pour s'assurer que les deux sont chargés dans le même ordre
# image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
t1_files = sorted([f for f in os.listdir(image_dirs["T1"]) if f.endswith('.npy')])
t2_files = sorted([f for f in os.listdir(image_dirs["T2"]) if f.endswith('.npy')])
t1ce_files = sorted([f for f in os.listdir(image_dirs["T1ce"]) if f.endswith('.npy')])
flair_files = sorted([f for f in os.listdir(image_dirs["Flair"]) if f.endswith('.npy')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

images = []
masks = []

# Charger toutes les images et leurs masques / Version pour un seul canal
# for image_file, mask_file in zip(image_files, mask_files):
#     img, mask = load_image_mask(os.path.join(image_dir, image_file), os.path.join(mask_dir, mask_file))
#     images.append(img)
#     masks.append(mask)

# Charger toutes les images et leurs masques
for t1_file, t2_file, t1ce_file, flair_file, mask_file in zip(t1_files, t2_files, t1ce_files, flair_files, mask_files):
    img, mask = load_multimodal_images(
        os.path.join(image_dirs["T1"], t1_file),
        os.path.join(image_dirs["T2"], t2_file),
        os.path.join(image_dirs["T1ce"], t1ce_file),
        os.path.join(image_dirs["Flair"], flair_file),
        os.path.join(mask_dir, mask_file)
    )
    images.append(img)
    masks.append(mask)

images = np.array(images)
masks = np.array(masks)

# Assurer la forme des données : (batch_size, height, width, channels)
# images = np.expand_dims(images, axis=-1)  # Ajouter une dimension pour les canaux (1 canal pour chaque image et masque) (240,240) -> (240,240,1), pas utile si plusieurs modalités
masks = np.expand_dims(masks, axis=-1)

# Séparer les données en ensembles de training et de validation
X_train, X_val, Y_train_0, Y_val_0 = train_test_split(images, masks, test_size=0.175, random_state=1)

# Convertir les labels en one-hot
Y_train = to_categorical(Y_train_0, num_classes=5)
Y_val = to_categorical(Y_val_0, num_classes=5)

# Test de la disponibilité du GPU
print("GPU dispo :", tf.config.list_physical_devices('GPU'))

#---------------------------------------------------------------------------------------#

# Création du modèle
model = Unet_plus_plus(input_shape)

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

#---------------------------------------------------------------------------------------#

# Sauvegarder le modèle
# model.save('unet_2D_best.keras')

# from IPython.display import FileLink
# print(os.listdir('/kaggle/working/'))
# os.chdir('/kaggle/working/')
# Téléchargez le modèle
# FileLink(r'unet_2D_best_model.keras')