import os
from dotenv import load_dotenv
from ultralytics import YOLO
from roboflow import Roboflow

# === 1. Налаштування ===
# Параметри

# Завантаження змінних з .env
load_dotenv()

# Зчитування параметрів
API_KEY = os.getenv("API_KEY")
WORKSPACE_NAME = "dekkan-neural-network-workspace" # Dekkan neural network workspace
PROJECT_NAME = "cats_detection-yjccj" # Cats_detection
DATASET_VERSION = 1
DATA_FORMAT = "yolov11"
MODEL_PATH = "models/yolo11m.pt"
TRAIN_EPOCHS = 25
IMG_SIZE = 640

# === 2. Завантаження датасету ===
def download_dataset():
    print("Завантаження датасету...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
    version = project.version(DATASET_VERSION)
    dataset = version.download(DATA_FORMAT)
    print(f"Датасет завантажено до: {dataset.location}")
    return dataset

# === 3. Тренування моделі ===
def train_model(dataset):
    print("Тренування моделі...")
    model = YOLO(MODEL_PATH)  # Завантаження предтренованої моделі

    print("Пристрій, що використовується:", model.device)
    model.train(
        data = f"{dataset.location}/data.yaml",
        epochs = TRAIN_EPOCHS,
        imgsz = IMG_SIZE,
        plots = True,
        save = True,
        project = 'runs/detect',  # Стандартний шлях
        name = 'train_result',  # Ім'я підкаталогу
        batch = 1,
    )
    print("Тренування завершено.")
    return model

# === 4. Основний блок ===
if __name__ == "__main__":
    print("Оберіть дію, що потрібно виконати:")
    print("1. Завантажити датасет")
    print("2. Завантаження датасету та тренування моделі")

    choice = input("Введіть номер вибору (1 або 2): ").strip()

    if choice == "1":
        # Завантаження датасету
        dataset = download_dataset()
    elif choice == "2":
        # Завантаження датасету
        dataset = download_dataset()
        
        # Тренування моделі
        model = train_model(dataset)

        # Отримання шляху до останньої тренувальної сесії
        runs_path = "runs/detect"
        latest_run = max([os.path.join(runs_path, d) for d in os.listdir(runs_path)], key=os.path.getmtime)
        print(f"Останній запуск збережено в: {latest_run}")

        print("Все завершено успішно.")
    else:
        print("Невірний вибір. Завершення програми.")
        exit(1)
   
    print("Все завершено успішно.")