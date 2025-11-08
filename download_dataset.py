
import shutil
from kagglehub import kagglehub
path = kagglehub.dataset_download('competitions/ieee-fraud-detection', force_download=True)
destination_path = 'data'
shutil.copytree(path, destination_path, dirs_exist_ok=True)