from core.config import dataset_path
from preprocessing.profiler import load_dataset, profile_dataset
from assistant.llm import analyze_dataset


profile = profile_dataset(load_dataset(dataset_path()))

print("\n========== DATASET PROFILE ==========\n")
print(profile)

print("\n========== LLM ANALYSIS ==========\n")

analysis = analyze_dataset(profile)

print(analysis)