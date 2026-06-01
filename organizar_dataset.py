from pathlib import Path
import random
import shutil


#nesse código eu organizei o dataset para o formato esperado
#  pelo YOLOv8-seg, criando as pastas necessárias para treino, validação e teste,
#  copiando as imagens e labels correspondentes e criando o arquivo data.yaml.

EXTENSOES_IMAGEM = [".jpg", ".jpeg", ".png", ".bmp"]

PASTA_BASE = Path("dataset")
PASTA_IMAGENS = PASTA_BASE / "images"
PASTA_LABELS = PASTA_BASE / "labels"
PASTA_SAIDA = Path("/content/dataset_yolo")

PROPORCAO_TREINO = 0.70
PROPORCAO_VALIDACAO = 0.20
SEMENTE = 42


def encontrar_pares():
    """Encontra imagens que possuem label correspondente."""
    pares = []

    for imagem in PASTA_IMAGENS.iterdir():
        if imagem.suffix.lower() not in EXTENSOES_IMAGEM:
            continue

        label = PASTA_LABELS / f"{imagem.stem}.txt"
        if label.exists():
            pares.append((imagem, label))
        else:
            print(f"Aviso: imagem sem label correspondente: {imagem.name}")

    return pares


def preparar_pastas():
    """Cria as pastas no formato esperado pelo YOLO."""
    if PASTA_SAIDA.exists():
        shutil.rmtree(PASTA_SAIDA)

    for conjunto in ["train", "val", "test"]:
        (PASTA_SAIDA / "images" / conjunto).mkdir(parents=True, exist_ok=True)
        (PASTA_SAIDA / "labels" / conjunto).mkdir(parents=True, exist_ok=True)


def copiar_pares(pares, conjunto):
    """Copia imagens e labels para o conjunto indicado."""
    total = len(pares)

    for i, (imagem, label) in enumerate(pares, start=1):
        shutil.copy(imagem, PASTA_SAIDA / "images" / conjunto / imagem.name)
        shutil.copy(label, PASTA_SAIDA / "labels" / conjunto / label.name)

        if i % 100 == 0 or i == total:
            print(f"{conjunto}: {i}/{total} pares copiados")


def gerar_yaml():
    """Cria o arquivo data.yaml usado pelo treinamento do YOLOv8."""
    conteudo = """path: /content/dataset_yolo
train: images/train
val: images/val
test: images/test

names:
  0: fissura
"""
    (PASTA_SAIDA / "data.yaml").write_text(conteudo, encoding="utf-8")


def main():
    if not PASTA_IMAGENS.exists() or not PASTA_LABELS.exists():
        raise FileNotFoundError(
            "Coloque o dataset em dataset/images e dataset/labels antes de executar."
        )

    pares = encontrar_pares()

    if not pares:
        raise RuntimeError("Nenhum par imagem/label foi encontrado.")

    random.seed(SEMENTE)
    random.shuffle(pares)

    total = len(pares)
    qtd_treino = int(total * PROPORCAO_TREINO)
    qtd_validacao = int(total * PROPORCAO_VALIDACAO)

    treino = pares[:qtd_treino]
    validacao = pares[qtd_treino:qtd_treino + qtd_validacao]
    teste = pares[qtd_treino + qtd_validacao:]

    preparar_pastas()
    copiar_pares(treino, "train")
    copiar_pares(validacao, "val")
    copiar_pares(teste, "test")
    gerar_yaml()

    print("\nDataset organizado com sucesso!")
    print(f"Total: {total}")
    print(f"Treino: {len(treino)}")
    print(f"Validação: {len(validacao)}")
    print(f"Teste: {len(teste)}")
    print("Arquivo criado: /content/dataset_yolo/data.yaml")


if __name__ == "__main__":
    main()
