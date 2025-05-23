# BrainSeg
Ce projet a pour but de réaliser la segmentation de tumeurs cérébrales (gliomes) à partir d'IRM en 3D à l'aide de modèles de Deep Learning. Un première étape consite à travailler sur des images 2D qui sont des slices extraites des ficheirs 3D. Tous les scripts sont écrits en Python et utilisent des bibliothèques telles que TensorFlow et Keras pour l'entraînement et la validation des modèles.
Un rapport explicatif au format PDF est disponible dans le répertoire.

## Prérequis

Ce projet nécessite les bibliothèques suivantes pour fonctionner correctement. Vous pouvez les installer facilement via **pip** ou utiliser un fichier `requirements.txt` pour l'installation automatique des dépendances.

Les principales bibliothèques utilisées sont :

- **os** : Pour interagir avec le système d'exploitation et gérer les chemins de fichiers
- **time** : Pour mesurer le temps d'exécution des différents processus
- **numpy** : Bibliothèque pour les calculs numériques et la manipulation des tableaux
- **matplotlib** : Bibliothèque pour créer des visualisations et graphiques
- **nibabel** : Pour charger et manipuler des fichiers au format NIfTI (.nii), couramment utilisé pour les données d'IRM
- **tensorflow** : Framework pour la création et l'entraînement de modèles de Deep Learning. Utilisé avec Keras pour la création de modèles de réseaux de neurones
- **scikit-learn** : Utilisé pour la séparation des données en ensembles d'entraînement et de test

## Structure des Fichiers

Voici une description des fichiers inclus dans ce dépôt ainsi que leur fonction :

### `train_2d.py`

Ce fichier contient le code nécessaire pour entraîner des modèles de Deep Learning à la segmentation d'IRM en 2D. Le modèle est entraîné sur des images 2D extraites de volumes d'IRM 3D, et il permet de segmenter des tumeurs cérébrales. Le fichier inclut des étapes telles que la préparation des données, la création du modèle, l'entraînement, l'afficahe des courbes de précisions et la sauvegarde du modèle final.

### `test_2d.py`

Ce fichier permet de charger un modèle préalablement entraîné et de tester ses performances sur un dataset de test. Il permet aussi de calculer des métriques de performance telles que les scores IoU (Intersection over Union) et Dice et d'afficher une matrice de confusion pour évaluer la qualité de la segmentation.

### `extracteur.py`

Ce script est utilisé pour extraire la slice médiane (ou d'une autre) d'un volume d'IRM 3D, afin de réduire les données en images 2D.

### `display.py`

Ce fichier permet de charger et d'afficher une image IRM 2D ainsi que son masque de segmentation associé. C'est utile pour visualiser les données du dataset.

### `tri.py`

Le rôle de ce fichier est de choisir au hasard les fichiers (dossiers de patients) à utiliser pour constituer les datasets d'entraînement et de test. Il permet de séparer les données de manière aléatoire afin d'éviter tout biais dans la sélection des échantillons.

### `logger.py`

Ce fichier configure les logs pour suivre le déroulement de l'exécution du projet. Il utilise la bibliothèque `logging` et `colorlog` pour personnaliser les couleurs des logs selon le niveau d'importance.

## Datasets

### Dataset 2D

Le dataset 2D utilisé pour l'entraînement et le test est constitué de **slices médianes extraites de volumes d'IRM 3D**.

Ce dataset 2D est disponible sur Kaggle :
- [Dataset 2D - 2D Brain](https://www.kaggle.com/datasets/lonidasspartiate/2d-brain)

Le dataset a été divisé en un ensemble de **training** (85%) et un ensemble de **test** (15%).

### Dataset 3D

Le dataset 3D est le dataset d'entraînement **BraTS 2020**, qui contient des volumes d'IRM cérébrales en 3D annotés, utilisés pour la segmentation des tumeurs cérébrales. Le dataset de validation BraTS2020 est disponible publiquement mais pas la vérité terrain (les masques de segmentation) associée, il n'est donc pas utilisé dans ce projet. Le dataset de test n'est pas disponible publiquement.

Ce dataset 3D est disponible sur Kaggle :
- [Dataset 3D - BRATS 2020](https://www.kaggle.com/datasets/lonidasspartiate/brats2020-3d)

Ce dataset a également été divisé en un ensemble de **training** (85%) et un ensemble de **test** (15%).

