from pathlib import Path
import cv2
import numpy as np

#nesse codigo eu criei uma funcao que recebe um caminho 
#de imagem e um caminho de label e desenha os pontos da 
#label na imagem
def desenhar_label(caminho_imagem, caminho_label, caminho_saida="saida/label_visualizado.jpg"):
    caminho_imagem = Path(caminho_imagem)
    caminho_label = Path(caminho_label)
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    imagem = cv2.imread(str(caminho_imagem))
    if imagem is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

    altura, largura = imagem.shape[:2]
    mascara = np.zeros_like(imagem)

    linhas = caminho_label.read_text(encoding="utf-8").splitlines()

    for linha in linhas:
        partes = linha.strip().split()
        if len(partes) < 7:
            continue

        coords = list(map(float, partes[1:]))
        pontos = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * largura)
            y = int(coords[i + 1] * altura)
            pontos.append([x, y])

        pontos = np.array(pontos, dtype=np.int32)
        cv2.fillPoly(mascara, [pontos], (0, 0, 255))
        cv2.polylines(imagem, [pontos], True, (0, 255, 255), 2)

    resultado = cv2.addWeighted(imagem, 0.75, mascara, 0.25, 0)
    cv2.imwrite(str(caminho_saida), resultado)
    print(f"Imagem salva em: {caminho_saida}")


if __name__ == "__main__":
    print("Edite este arquivo ou importe a função desenhar_label() para visualizar um par imagem/label.")
