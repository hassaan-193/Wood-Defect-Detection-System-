import time
from bibliometrics import run_bibliometric_analysis
from vision_pipeline import run_vision_comparison

def main():
    print("\n" + "=" * 35)
    print("  WOOD SURFACE DEFECT DETECTION PIPELINE")
    print("  Based on: Goycochea Casas et al., Forests 2026, 17, 112")
    print("=" * 35)

    total_start = time.time()

    # 1. Bibliometric Analysis & Chart Generation
    try:
        run_bibliometric_analysis()
    except Exception as e:
        print(f"\nStage 1 (Bibliometrics) failed: {e}")

    # 2. Computer Vision Pipeline Comparison
    try:
        run_vision_comparison(n_samples=3)
    except Exception as e:
        print(f"\nStage 2 (Vision Pipeline) failed: {e}")

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 35)
    print(f"  PIPELINE COMPLETE in {total_elapsed/60:.1f} minutes")
    print()
    print("  To view the results, check the 'outputs' folder.")
    print("=" * 35 + "\n")

if __name__ == "__main__":
    main()
