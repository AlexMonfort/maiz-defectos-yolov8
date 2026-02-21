import os
import yaml

def check_dataset(yaml_path='data/dataset.yaml'):
    if not os.path.exists(yaml_path):
        print(f"Error: No se encontró {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Rutas a verificar
    paths = ['train', 'val', 'test']
    for p in paths:
        if p in data:
            # YOLO yaml paths are often relative to the yaml file or the root
            # We'll check relative to the current working directory
            # adjusted for the '../' in our specific yaml
            full_path = os.path.join('data', data[p])
            if os.path.exists(full_path):
                num_imgs = len([f for f in os.listdir(full_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"OK: {p} existe en {full_path} con {num_imgs} imágenes.")
            else:
                print(f"ADVERTENCIA: La ruta {p} ({full_path}) no existe.")

if __name__ == '__main__':
    check_dataset()
