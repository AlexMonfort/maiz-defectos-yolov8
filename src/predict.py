from ultralytics import YOLO
import argparse
import cv2

def predict(source, weights='runs/detect/maiz_defectos_training/weights/best.pt'):
    model = YOLO(weights)
    
    # Realizar inferencia
    results = model.predict(source=source, save=True, conf=0.25)
    
    for r in results:
        print(f"Detecciones en {r.path}: {len(r.boxes)} objetos encontrados.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Ruta a la imagen o video')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Ruta a los pesos del modelo')
    args = parser.parse_args()
    
    predict(args.source, args.weights)
