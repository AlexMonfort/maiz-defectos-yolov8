from ultralytics import YOLO
import cv2
import numpy as np

def calculate_severity(image_path, weights='runs/detect/maiz_defectos_training/weights/best.pt'):
    """
    Calcula la severidad basada en el área de los defectos detectados (Pseudo-segmentación).
    """
    model = YOLO(weights)
    results = model.predict(source=image_path, conf=0.25)
    
    img = cv2.imread(image_path)
    total_area = img.shape[0] * img.shape[1]
    defect_area = 0
    
    for r in results:
        for box in r.boxes:
            # Obtener coordenadas [x1, y1, x2, y2]
            coords = box.xyxy[0].tolist()
            w = coords[2] - coords[0]
            h = coords[3] - coords[1]
            defect_area += (w * h)
            
            # Aquí se podría añadir lógica de segmentación de color dentro del box
            # para una "pseudo-segmentación" más precisa.
            
    severity_idx = (defect_area / total_area) * 100
    print(f"Severidad estimada: {severity_idx:.2f}% de área afectada.")
    return severity_idx

if __name__ == '__main__':
    # Ejemplo de uso
    # calculate_severity('assets/demo/example_input.jpg')
    print("Módulo de cálculo de severidad cargado.")
