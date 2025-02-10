import os
import random
import shutil
from logger import logger

# Chemin du dataset
dataset_path = "H:\Marc\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\Train"
test_path = "H:\Marc\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\Test"

# Créer le sous-dossier s'il n'existe pas
os.makedirs(test_path, exist_ok=True)

# Liste des numéros des dossiers contenant les IRM (1 à 369)
patient_ids = [f"{i:03d}" for i in range(1, 370)]

# Sélectionner 55 numéros aléatoires
random.seed(0) # si l'on veut reproduire le même tirage
test_patient_ids = random.sample(patient_ids, 55)
test_patient_ids.sort() # tri des numéros dans l'ordre croissant

# Déplacer les dossiers correspondants dans le dossier 'Test'
for patient_id in test_patient_ids:
    source = os.path.join(dataset_path, f"BraTS20_Training_{patient_id}")
    destination = os.path.join(test_path, f"BraTS20_Training_{patient_id}")
    shutil.move(source, destination)

# Sauvegarder la liste des patients pour la phase de test dans un fichier texte
with open("test_numeros.txt", "w") as f:
    for patient_id in test_patient_ids:
        f.write(f"BraTS20_Training_{patient_id}\n")

logger.info("Liste des dossiers utilisés pour la phase de test enregistrée dans 'test_numeros.txt'")