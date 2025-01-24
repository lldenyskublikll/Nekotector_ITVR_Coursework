import os
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2

# === 1. Налаштування ===
# Завантаження змінних з .env
load_dotenv()

# Константи
PRETRAINED_MODEL_PATH = "models/yolov11m.pt"  # Предтренована модель
TRAINED_MODEL_DIR = "runs/detect"  # Директорія з натренованими моделями

TEST_IMAGES_DIR = "C:/Users/ddkub/Desktop/ITVR_Coursework/test_images"  # Папка з тестовими зображеннями
#TEST_IMAGES_DIR = "C:/Users/ddkub/Desktop/ITVR_Coursework/test_images_real"  # Папка з власними тестовими зображеннями
TEST_VIDEOS_DIR = "C:/Users/ddkub/Desktop/ITVR_Coursework/test_videos"  # Папка з тестовими відео

#LATEST_MODEL = "runs/detect/train_result"
LATEST_MODEL = "researches/yolo11m_(25_epochs,_16_img_per_batch)/runs/detect/train_result"
#LATEST_MODEL = "researches/yolo11m_(50_epochs,_16_img_per_batch)/runs/detect/train_result"
#LATEST_MODEL = "researches/yolo11m_(100_epochs,_16_img_per_batch)/runs/detect/train_result"


# === 2. Завантаження моделі ===
def load_model(trained: bool):
    """
    Завантажує натреновану або предтреновану модель.
    
    :param trained: Якщо True, використовується остання натренована модель.
    :return: Завантажена модель YOLO.
    """
    if trained:
        # Знайти останню натреновану модель
        if not os.path.exists(TRAINED_MODEL_DIR):
            raise FileNotFoundError(f"Директорія з натренованими моделями не знайдена: {TRAINED_MODEL_DIR}")
        
        print(f"Використовується натренована модель: {LATEST_MODEL}")
        return YOLO(os.path.join(LATEST_MODEL, "weights/best.pt"))
    else:
        print("Використовується предтренована модель.")
        return YOLO(PRETRAINED_MODEL_PATH)
    
# === 3. Тестування моделі на зображеннях ===
def test_model_on_folder(model, folder_path):
    """
    Виконує передбачення для всіх зображень у вказаній папці.
    
    :param model: Модель YOLO для тестування.
    :param folder_path: Шлях до папки із зображеннями для тестування.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Папка з тестовими зображеннями не знайдена: {folder_path}")

    print(f"Тестування моделі на зображеннях з папки: {folder_path}...")
    results = model.predict(source=folder_path, save=True)
    print("Детекцію завершено. Результати збережено у папці runs/predict.")
    
# === 4. Тестування моделі на відео ===
def test_model_on_video(model, folder_path):
    """
    Виконує передбачення для всіх відео у вказаній папці.

    :param model: Модель YOLO для тестування.
    :param folder_path: Шлях до папки із відео для тестування.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Папка з тестовими відео не знайдена: {folder_path}")

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("У папці немає відеофайлів для тестування.")
        return

    output_dir = os.path.join("runs", "predict_videos")
    os.makedirs(output_dir, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        output_path = os.path.join(output_dir, f"predicted_{video_file}")

        print(f"Обробка відео: {video_path}...")

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, save=False, conf=0.5)

            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        print(f"Обробка завершена. Відео збережено за адресою: {output_path}")

    print("Детекцію завершено. Результати збережено у папці runs/predict_videos.")
    
# === 5. Основний блок ===
if __name__ == "__main__":
    print("Оберіть модель для тестування:")
    print("1. Натренована модель")
    print("2. Предтренована модель")

    choice = input("Введіть номер вибору (1 або 2): ").strip()
    mode = 0

    if choice == "1":
        mode = 1
        try:
            model = load_model(trained=True)
        except FileNotFoundError as e:
            print(e)
            print("Перевірте наявність натренованих моделей і повторіть спробу.")
            exit(1)
    elif choice == "2":
        mode = 0   
        model = load_model(trained=False)
    else:
        print("Невірний вибір. Завершення програми.")
        exit(1)

    if mode == 1:
        print("Оберіть тип даних для тестування моделі:")
        print("1. Зображення")
        print("2. Відео")
        
        choice_1 = input("Введіть номер вибору (1 або 2): ").strip()
        
        if choice_1 == "1":
            # Тестування обраної моделі на папці зображень
            test_model_on_folder(model, TEST_IMAGES_DIR)
        elif choice_1 == "2":
            # Тестування обраної моделі на відео
            test_model_on_video(model, TEST_VIDEOS_DIR)
        else:
            print("Невірний вибір. Завершення програми.")
            exit(1)
        
    else:
        # Тестування обраної моделі на папці зображень
        test_model_on_folder(model, TEST_IMAGES_DIR)
    
    print("Все завершено успішно.")