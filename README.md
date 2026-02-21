# Maíz Defectos YOLOv8

Este proyecto utiliza YOLOv8 para la detección y clasificación de defectos en granos de maíz.

## Estructura del Proyecto

- `data/`: Contiene la configuración del dataset.
- `src/`: Scripts de entrenamiento, validación y predicción.
- `src/dataset_tools/`: Herramientas para limpieza y verificación de datos.
- `assets/`: Imágenes de demostración y resultados.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Entrenamiento
```bash
python src/train.py
```

### Predicción
```bash
python src/predict.py --source path/to/image.jpg
```
