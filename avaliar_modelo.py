from ultralytics import YOLO

#nesse código eu avaliei o modelo de detecção de fissuras usando o método `val` da classe `YOLO`.

modelo = YOLO("modelos/best.pt")

metricas = modelo.val(
    data="/content/dataset_yolo/data.yaml",
    split="test",
    imgsz=640,
    device=0
)

print(metricas)