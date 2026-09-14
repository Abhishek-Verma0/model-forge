from core.config import dataset_path
from preprocessing.profiler import load_dataset, profile_dataset


profile = profile_dataset(load_dataset(dataset_path()))

print(profile)