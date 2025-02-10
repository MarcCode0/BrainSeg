import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Charger un fichier IRM (exemple : séquence T2)
path_nii = "H:\Marc\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii"
img_nii = nib.load(path_nii)

# Convertir en tableau numpy
img_data = img_nii.get_fdata()

# Sélectionner une coupe au centre (axe axial)
slice_idx = img_data.shape[2] // 2
img_slice = img_data[:, :, slice_idx]

# Affichage
plt.imshow(img_slice, cmap='gray')
plt.title("IRM T2 - Coupe axiale")
plt.axis("off")
plt.show()

# Charger le fichier masque (annotation de la tumeur)
path_mask = "H:\Marc\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii"
mask_nii = nib.load(path_mask)
mask_data = mask_nii.get_fdata()

# Même coupe que l’IRM
mask_slice = mask_data[:, :, slice_idx]

# Affichage avec superposition du masque
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_slice, cmap='gray')
plt.title("IRM T2 - Coupe axiale")

plt.subplot(1, 2, 2)
plt.imshow(img_slice, cmap='gray')
plt.imshow(mask_slice, cmap='jet', alpha=0.5)  # Superposition du masque en couleur
plt.title("IRM avec segmentation de la tumeur")
plt.show()

# Effectuer une rotation de l'image pour corriger l'orientation
# img_slice_b = img_data[120,:,:]
# img_slice_b_rotated = np.rot90(img_slice_b, k=1)  # k= 1 fait une rotation de 90° dans le sens inverse des aiguilles d'une montre