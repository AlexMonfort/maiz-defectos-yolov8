from ultralytics import YOLO

def validate_model(weights_path='runs/detect/maiz_defectos_training/weights/best.pt'):
    # Cargar el modelo entrenado
    model = YOLO(weights_path)

    # Ejecutar validación
    metrics = model.val(data='data/dataset.yaml')
    
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")

if __name__ == '__main__':
    validate_model()
