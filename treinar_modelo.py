from pathlib import Path
import shutil
from ultralytics import YOLO

DATA_YAML = Path("/content/dataset_yolo/data.yaml")
PASTA_MODELOS = Path("modelos")

#já nesse código eu configurei o treinamento do modelo de detecção 
#de fissuras usando o YOLOv8-seg. O script verifica se o arquivo 
#data.yaml existe, inicia o treinamento com os parâmetros definidos 
#e, ao final, salva o melhor modelo encontrado na pasta "modelos/best.pt". 
#O código é estruturado para ser executado em um ambiente com GPU, como o Google Colab.

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError("Execute primeiro: python organizar_dataset.py")

    modelo = YOLO("yolov8n-seg.pt")

    modelo.train(
        data=str(DATA_YAML),
        epochs=60,
        imgsz=640,
        batch=16,
        patience=15,
        device=0,
        project="runs_fissuramap",
        name="yolov8n_seg_fissuras",
        exist_ok=True,
    )

    melhor_modelo = Path("runs_fissuramap/yolov8n_seg_fissuras/weights/best.pt")

    if melhor_modelo.exists():
        PASTA_MODELOS.mkdir(exist_ok=True)
        destino = PASTA_MODELOS / "best.pt"
        shutil.copy(melhor_modelo, destino)
        print(f"Modelo salvo em: {destino}")
    else:
        print("Treinamento finalizado, mas o arquivo best.pt não foi encontrado.")


if __name__ == "__main__":
    main()
