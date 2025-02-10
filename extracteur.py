import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from logger import logger
from tqdm import tqdm  # Barre de progression

# Se placer dans le répertoire de travail
global_path = "H:\Marc\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
os.chdir(global_path)

# Dossiers
input_folder = "Train"  # Contient les dossiers patients
output_folder = "slices_2D_Flair"  # Où sauvegarder les images extraites

# Extraction des slices médianes
def extracteur_slice_med(input_folder, output_folder, file_suffix):
    """
    Extrait la slice médiane de fichiers NIfTI (.nii) et les enregistre en PNG.

    - input_folder: Dossier contenant les dossiers des patients
    - output_folder: Dossier où enregistrer les slices extraites
    - file_suffix: Suffixe du fichier à extraire ("flair", "seg", "t1", "t1ce", "t2")
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Liste des patients
    patients = sorted(os.listdir(input_folder))  # Tri par n° croissant

    for patient in tqdm(patients, desc=f"Extraction des slices {file_suffix}"):
        patient_path = os.path.join(input_folder, patient)
        file_path = os.path.join(patient_path, f"{patient}_{file_suffix}.nii")

        if os.path.exists(file_path):  # Vérifier si le fichier existe
            img = nib.load(file_path).get_fdata()  # Charger le volume en numpy array

            # Slice médiane
            z_mid = img.shape[2] // 2
            slice_med = img[:, :, z_mid]

            output_path = os.path.join(output_folder, f"{patient}_slice_{file_suffix}.npy")

            if file_suffix == "seg":
                np.save(output_path, slice_med.astype(np.uint8))  # Valeurs 0,1,2,4
            else:
                np.save(output_path, slice_med.astype(np.float32))
        else:
            logger.warning(f"Fichier non trouvé pour le patient {patient} ({file_suffix})")

    logger.info(f"Extraction terminée pour les fichiers {file_suffix}")