import sys
import time
from pathlib import Path

#This file is used to run the preprocessing ablation study, 
#to show empirical results of the preprocessing pipeline.

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third.processing import EmotionImageProcessor
from third.predict import EmotionPredictor


DATASET_PATH = "data/facial-emotion-recognition"
OUTPUT_ROOT = Path("data/facial-emotion-recognition-preprocessed-ablation")

ABLATION_TESTS = [
    {
        "name": "Baseline",
        "description": "Grayscale only",
        "preprocess_kwargs": {
            "grayscale": True,
            "align": False,
            "crop": False,
            "normalize": False,
        },
    },
    {
        "name": "+Alignment",
        "description": "Baseline + alignment",
        "preprocess_kwargs": {
            "grayscale": True,
            "align": True,
            "crop": False,
            "normalize": False,
        },
    },
    {
        "name": "+Crop",
        "description": "Alignment + cropping",
        "preprocess_kwargs": {
            "grayscale": True,
            "align": True,
            "crop": True,
            "normalize": False,
        },
    },
    {
        "name": "+Normalizeation",
        "description": "Alignment + cropping + normalization",
        "preprocess_kwargs": {
            "grayscale": True,
            "align": True,
            "crop": True,
            "normalize": True,
        },
    },
]


def run_single_experiment(test_name: str, preprocess_kwargs: dict) -> dict:
    start = time.perf_counter()

    processor = EmotionImageProcessor(DATASET_PATH)

    output_dir = OUTPUT_ROOT / test_name.lower()
    processor.preprocess(save_dir=output_dir, **preprocess_kwargs)

    processor.train_val_test_split(val_size=0.2, test_size=0.2, random_state=42)

    lbp_train, lbp_val, lbp_test = processor.apply_lbp()

    processor.X_train = lbp_train
    processor.X_val = lbp_val
    processor.X_test = lbp_test

    pca_train, pca_val, pca_test, _ = processor.apply_pca(n_components=50)

    predictor = EmotionPredictor(
        pca_train,
        processor.get_train_labels(),
        X_val=pca_val,
        y_val=processor.get_val_labels(),
        X_test=pca_test,
        y_test=processor.get_test_labels(),
    )

    predictor.train("random_forest", n_estimators=200)

    metrics = predictor.evaluate("random_forest", dataset="test")

    elapsed = time.perf_counter() - start

    return {
        "Test": test_name,
        "Accuracy": metrics["accuracy"],
        "Time (s)": elapsed,
    }


def run_ablation():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for test in ABLATION_TESTS:
        result = run_single_experiment(test["name"], test["preprocess_kwargs"])
        results.append(result)

    print("\nPreprocessing Ablation Study Results")
    print("-" * 60)
    print(f"{'Test':<10} {'Accuracy':>10} {'Time (s)':>12}")
    print("-" * 60)
    for result in results:
        print(
            f"{result['Test']:<10} {result['Accuracy']*100:>9.2f}% "
            f"{result['Time (s)']:>12.2f}"
        )
    print("-" * 60)


if __name__ == "__main__":
    run_ablation()

