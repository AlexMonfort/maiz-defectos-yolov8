from ultralytics import YOLO
import os

def train_model():
    # Cargar el modelo YOLOv8 nano (puedes usar 'yolov8s.pt', 'yolov8m.pt', etc.)
    model = YOLO('yolov8n.pt')

    # Entrenar el modelo
    results = model.train(
        data='data/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='maiz_defectos_training'
    )
    print("Entrenamiento completado. Resultados guardados en 'runs/detect/train'")

if __name__ == '__main__':
    train_model()
