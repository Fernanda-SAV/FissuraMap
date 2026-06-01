from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

#nesse código eu criei a classe `FissuraMap`, que é responsável por carregar 
#o modelo de detecção de fissuras e analisar as imagens 
#enviadas pelo usuário. A classe possui métodos para descrever a
#posição, orientação e extensão das fissuras, além de classificar o nível de atenção 
#com base nas métricas calculadas. O método `analisar` é o principal, onde ocorre a 
#detecção e análise das fissuras na imagem fornecida.

class FissuraMap:
    def __init__(self, caminho_modelo="modelos/best.pt", confianca_minima=0.25):
        self.caminho_modelo = Path(caminho_modelo)
        self.confianca_minima = confianca_minima

        if not self.caminho_modelo.exists():
            raise FileNotFoundError(
                "Modelo não encontrado. Treine o modelo primeiro e salve em modelos/best.pt."
            )

        self.modelo = YOLO(str(self.caminho_modelo))

    def analisar(self, caminho_imagem):
        caminho_imagem = Path(caminho_imagem)
        imagem = cv2.imread(str(caminho_imagem))

        if imagem is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")

        altura, largura = imagem.shape[:2]

        predicoes = self.modelo.predict(
            source=str(caminho_imagem),
            conf=self.confianca_minima,
            imgsz=640,
            verbose=False,
        )

        resultado = predicoes[0]
        imagem_marcada = imagem.copy()
        mascara_total = np.zeros((altura, largura), dtype=np.uint8)
        fissuras = []

        if resultado.masks is None:
            return {
                "quantidade": 0,
                "confianca_media": 0.0,
                "area_total_percentual": 0.0,
                "nivel_atencao": "baixo",
                "mensagem": "Nenhuma fissura foi localizada com a confiança mínima selecionada.",
                "imagem_marcada": imagem_marcada,
                "mascara_total": mascara_total,
                "fissuras": [],
            }

        mascaras = resultado.masks.data.cpu().numpy()
        confiancas = resultado.boxes.conf.cpu().numpy()

        for indice, mascara in enumerate(mascaras):
            mascara_redimensionada = cv2.resize(mascara, (largura, altura))
            mascara_binaria = (mascara_redimensionada > 0.5).astype(np.uint8) * 255
            mascara_total = cv2.bitwise_or(mascara_total, mascara_binaria)

            contornos, _ = cv2.findContours(
                mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contornos:
                continue

            contorno = max(contornos, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contorno)
            area_pixels = cv2.countNonZero(mascara_binaria)
            area_percentual = 100 * area_pixels / (altura * largura)
            confianca = float(confiancas[indice]) * 100

            posicao = self._descrever_posicao(x, y, w, h, largura, altura)
            orientacao = self._descrever_orientacao(w, h)
            extensao = self._estimar_extensao(w, h, largura, altura)

            fissuras.append(
                {
                    "id": indice + 1,
                    "confianca": round(confianca, 1),
                    "area_percentual": round(area_percentual, 3),
                    "posicao": posicao,
                    "orientacao": orientacao,
                    "extensao_relativa": extensao,
                    "caixa": (int(x), int(y), int(w), int(h)),
                }
            )

            camada = imagem_marcada.copy()
            camada[mascara_binaria > 0] = (0, 0, 255)
            imagem_marcada = cv2.addWeighted(camada, 0.40, imagem_marcada, 0.60, 0)

            cv2.rectangle(imagem_marcada, (x, y), (x + w, y + h), (0, 180, 255), 2)
            cv2.putText(
                imagem_marcada,
                f"Fissura {indice + 1}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 180, 255),
                2,
            )

        quantidade = len(fissuras)
        area_total = 100 * cv2.countNonZero(mascara_total) / (altura * largura)
        confianca_media = float(np.mean(confiancas)) * 100 if quantidade > 0 else 0.0
        nivel = self._classificar_atencao(area_total, quantidade, fissuras)
        mensagem = self._gerar_mensagem(nivel, quantidade)

        return {
            "quantidade": quantidade,
            "confianca_media": round(confianca_media, 1),
            "area_total_percentual": round(area_total, 3),
            "nivel_atencao": nivel,
            "mensagem": mensagem,
            "imagem_marcada": imagem_marcada,
            "mascara_total": mascara_total,
            "fissuras": fissuras,
        }

    def _descrever_posicao(self, x, y, w, h, largura, altura):
        centro_x = x + w / 2
        centro_y = y + h / 2

        if centro_y < altura / 3:
            faixa_y = "superior"
        elif centro_y < 2 * altura / 3:
            faixa_y = "central"
        else:
            faixa_y = "inferior"

        if centro_x < largura / 3:
            faixa_x = "esquerda"
        elif centro_x < 2 * largura / 3:
            faixa_x = "central"
        else:
            faixa_x = "direita"

        if faixa_y == "central" and faixa_x == "central":
            return "região central da parede"

        return f"região {faixa_y} {faixa_x} da parede"

    def _descrever_orientacao(self, w, h):
        proporcao = w / max(h, 1)

        if proporcao >= 1.8:
            return "predominantemente horizontal"
        if proporcao <= 0.55:
            return "predominantemente vertical"
        return "irregular ou ramificada"

    def _estimar_extensao(self, w, h, largura, altura):
        diagonal_fissura = np.sqrt(w ** 2 + h ** 2)
        diagonal_imagem = np.sqrt(largura ** 2 + altura ** 2)
        return round(100 * diagonal_fissura / diagonal_imagem, 2)

    def _classificar_atencao(self, area_total, quantidade, fissuras):
        maior_extensao = max([f["extensao_relativa"] for f in fissuras], default=0)

        if quantidade == 0:
            return "baixo"
        if area_total >= 2.0 or quantidade >= 5 or maior_extensao >= 60:
            return "crítico"
        if area_total >= 1.0 or quantidade >= 3 or maior_extensao >= 35:
            return "alto"
        if area_total >= 0.35 or quantidade >= 2:
            return "moderado"
        return "baixo"

    def _gerar_mensagem(self, nivel, quantidade):
        if quantidade == 0:
            return "Nenhuma fissura foi localizada na imagem analisada."
        if nivel == "crítico":
            return "Fissuras relevantes foram localizadas. Recomenda-se inspeção técnica antes da pintura ou revestimento."
        if nivel == "alto":
            return "Foram localizadas fissuras com atenção elevada. Recomenda-se verificar a região marcada antes das próximas etapas da obra."
        if nivel == "moderado":
            return "Foram localizadas fissuras pontuais. Recomenda-se correção preventiva antes do acabamento."
        return "Fissura localizada com baixo nível de atenção, mantendo recomendação de registro e acompanhamento."
