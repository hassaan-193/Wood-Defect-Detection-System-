import os

# 1. PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
BIBLIOMETRIC_FILE = os.path.join(BASE_DIR, "data_clean.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")

# 2. DATASET SETTINGS
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
IMG_SIZE = 640

# 3. YOLOV8 TRAINING SETTINGS
YOLO_MODEL = "yolov8n.pt"
EPOCHS       = 50
BATCH_SIZE   = 16
LEARNING_RATE = 0.01
PATIENCE     = 10
WORKERS      = 4
TRAIN_OUTPUT = os.path.join(MODELS_DIR, "yolov8_wood_defect")

# 4. CLASSICAL CV SETTINGS
CANNY_LOW  = 50
CANNY_HIGH = 150
MIN_CONTOUR_AREA = 500
MORPH_KERNEL_SIZE = 5

# 5. DEFECT CLASS NAMES
DEFECT_CLASSES = [
    "crack",
    "knot",
    "stain",
    "scratch",
]

# 6. COLORS FOR VISUALIZATION
CLASS_COLORS = [
    (0,   0,   255),   # Red -> crack
    (0,   255,  0),    # Green -> knot
    (255, 165,  0),    # Orange -> stain
    (255,  0,  255),   # Purple -> scratch
    (0,   255, 255),   # Yellow -> extra class
    (255, 255,  0),    # Cyan  -> extra class
]

# 7. CREATE FOLDERS AUTOMATICALLY
for folder in [MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, RESULTS_DIR,
               os.path.join(BASE_DIR, "data")]:
    os.makedirs(folder, exist_ok=True)

print("Config loaded. Project folders ready.")
