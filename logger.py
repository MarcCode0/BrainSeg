import logging
import colorlog

formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',  # Format de l'heure
    log_colors={  # Définition des couleurs pour chaque niveau de log
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

# Configuration du handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Création du logger global
logger = logging.getLogger("my_logger")  # Nom du logger
logger.setLevel(logging.DEBUG)  # Niveau minimum des logs

# Ajout du handler au logger
logger.addHandler(handler)

# Empêcher la duplication des logs si on importe plusieurs fois
if not logger.hasHandlers():
    logger.addHandler(handler)