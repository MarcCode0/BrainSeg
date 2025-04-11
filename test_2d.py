import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

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
global_path = "/kaggle/input/"
os.chdir(global_path)

# Charger un modèle
model = load_model('u-net-pp-2d-batchnormalization/keras/default/2/unet_pp_2D_1024_BN_2.keras')

global_path = "/kaggle/input/2d-brain/"
os.chdir(global_path)
# Dossiers des images 2D (slices) et des masques de segmentation
# image_dir = "test_slices_2D_Flair"
image_dirs = {
    "T1": "test_slices_2D_T1",
    "T2": "test_slices_2D_T2",
    "T1ce": "test_slices_2D_T1ce",
    "Flair": "test_slices_2D_Flair"
}
mask_dir = "test_slices_2D_Seg"

# Listes des fichiers d'images et de masques (fichiers .npy) pour s'assurer que les deux sont chargés dans le même ordre
# image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
t1_files = sorted([f for f in os.listdir(image_dirs["T1"]) if f.endswith('.npy')])
t2_files = sorted([f for f in os.listdir(image_dirs["T2"]) if f.endswith('.npy')])
t1ce_files = sorted([f for f in os.listdir(image_dirs["T1ce"]) if f.endswith('.npy')])
flair_files = sorted([f for f in os.listdir(image_dirs["Flair"]) if f.endswith('.npy')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

X_test = []
masks = []

# Charger toutes les images et leurs masques
# for image_file, mask_file in zip(image_files, mask_files):
#     img, mask = load_image_mask(os.path.join(image_dir, image_file), os.path.join(mask_dir, mask_file))
#     X_test.append(img)
#     masks.append(mask)

for t1_file, t2_file, t1ce_file, flair_file, mask_file in zip(t1_files, t2_files, t1ce_files, flair_files, mask_files):
    img, mask = load_multimodal_images(
        os.path.join(image_dirs["T1"], t1_file),
        os.path.join(image_dirs["T2"], t2_file),
        os.path.join(image_dirs["T1ce"], t1ce_file),
        os.path.join(image_dirs["Flair"], flair_file),
        os.path.join(mask_dir, mask_file)
    )
    X_test.append(img)
    masks.append(mask)

X_test = np.array(X_test)
Y_test = np.array(masks)

# Convertir les labels en one-hot
Y_test_oh = to_categorical(Y_test, num_classes=5)

# Évaluer le modèle sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(X_test, Y_test_oh, verbose=1)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

#---------------------------------------------------------------------------------------#

# Effectuer des prédictions sur l'ensemble de validation
# predictions = model.predict(X_test)
# predictions = model(X_test, training=False)  # Force l'utilisation des statistiques globales de BatchNorm

# Charger une image de test et la prédiction correspondant
id_img = 0
# img_input = X_test[id_img]  # Image d'entrée
img_input = X_test[id_img,:,:,3]  # Image d'entrée
img_pred = np.argmax(predictions, axis=-1)[id_img]  # Prédiction (masque)

# Afficher l'image d'entrée (en niveaux de gris)
plt.imshow(img_input, cmap='gray')

# Afficher le masque prédit par-dessus l'image avec un peu de transparence (alpha)
plt.imshow(img_pred, cmap='afmhot', vmin =0, vmax = 4, alpha=0.5)  # alpha contrôle la transparence du masque (0: transparent, 1: opaque)
plt.title(f"Masque prédit (image de test {id_img})")
plt.show()

# Afficher le masque de la vérité terrain (réel)
plt.imshow(np.argmax(Y_test_oh[id_img], axis=-1), cmap='afmhot', vmin =0, vmax = 4,)  # Afficher le masque réel
plt.title(f"Masque réel (image de test {id_img})")
plt.show()

#---------------------------------------------------------------------------------------#

from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm  # Import pour l'échelle logarithmique
import seaborn as sns

# Suppresion de la classe 3
predictions2 = np.delete(predictions, 3, axis=-1)  # Shape devient (55, 240, 240, 4)
Y_test_oh2 = np.delete(Y_test_oh, 3, axis=-1)


# Afficher la matrice de confusion
def plot_confusion_matrix(y_true, y_pred, labels):
    y_true = np.argmax(y_true, axis=-1).flatten()
    y_pred = np.argmax(y_pred, axis=-1).flatten()
    y_true[y_true == 3] = 4
    y_pred[y_pred == 3] = 4

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, norm=LogNorm())
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.title("Matrice de confusion")
    plt.show()

# Afficher la matrice de confusion
plot_confusion_matrix(Y_test_oh2, predictions2, [0, 1, 2, 4])

#---------------------------------------------------------------------------------------#
# Définition des scores IoU (intersection over union) et Dice

def iou_score(y_true, y_pred, num_classes=4):
    y_true = np.argmax(y_true, axis=-1)  # Convertir one-hot en labels
    y_pred = np.argmax(y_pred, axis=-1)

    iou_per_class = []
    for c in range(num_classes):
        true_class = (y_true == c)
        pred_class = (y_pred == c)

        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()

        iou = intersection / union if union > 0 else np.nan  # Éviter la division par zéro
        iou_per_class.append(iou)

    return np.nanmean(iou_per_class), iou_per_class  # Moyenne globale et valeurs par classe
                                                     # np.nanmean calcule la moyenne en ignorant les valeurs NaN

def dice_score(y_true, y_pred, num_classes=4):
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    dice_per_class = []
    for c in range(num_classes):
        true_class = (y_true == c)
        pred_class = (y_pred == c)

        intersection = np.logical_and(true_class, pred_class).sum()
        dice = (2. * intersection) / (true_class.sum() + pred_class.sum()) if (true_class.sum() + pred_class.sum()) > 0 else np.nan
        dice_per_class.append(dice)

    return np.nanmean(dice_per_class), dice_per_class  # Moyenne globale et valeurs par classe

# Calcul de l'IoU sur l'ensemble de test
iou_moy, iou_classes = iou_score(Y_test_oh2, predictions2, num_classes=4)
print(f"IoU moyen: {iou_moy}")
print(f"IoU par classe: {iou_classes}")

# Calcul du Dice score
dice_moy, dice_classes = dice_score(Y_test_oh2, predictions2, num_classes=4)
print(f"Coefficient de Dice moyen: {dice_moy}")
print(f"Coefficient de Dice par classe: {dice_classes}")