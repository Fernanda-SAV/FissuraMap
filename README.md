# 🧱 FissuraMap

## Sistema Inteligente de Detecção e Localização de Fissuras em Dispositivos Móveis

### Desafio 2 – Bootcamp de Ciência de Dados e Inteligência Artificial

---

## 📌 Descrição

O **FissuraMap** é uma aplicação baseada em Inteligência Artificial e Visão Computacional desenvolvida para auxiliar processos de inspeção visual de paredes e estruturas de alvenaria.

A solução permite que um usuário capture uma imagem utilizando a câmera do dispositivo móvel ou selecione uma imagem da galeria. Em seguida, o sistema identifica automaticamente regiões contendo fissuras e trincas, destacando sua localização diretamente na imagem.

O projeto foi desenvolvido utilizando:

* YOLOv8 Segmentation
* Python
* OpenCV
* Streamlit

---

## 🎯 Objetivo

Automatizar a identificação e localização de fissuras em superfícies de alvenaria através de imagens capturadas por dispositivos móveis.

A solução foi projetada considerando:

* baixo custo computacional;
* compatibilidade com smartphones;
* facilidade de utilização;
* processamento rápido;
* interface amigável para inspeções em campo.

---

## 🚀 Funcionalidades

### Detecção automática

Identifica regiões contendo fissuras.

### Segmentação das fissuras

Utiliza máscaras de segmentação para representar o formato real da fissura.

### Localização espacial

Indica a posição aproximada da fissura na imagem:

* Superior esquerda
* Superior central
* Superior direita
* Centro esquerda
* Centro
* Centro direita
* Inferior esquerda
* Inferior central
* Inferior direita

### Classificação da orientação

O sistema identifica a orientação predominante:

* Horizontal
* Vertical
* Diagonal
* Irregular

### Indicador de atenção

Classifica automaticamente o nível de atenção:

* Baixo
* Moderado
* Alto
* Crítico

### Compatibilidade móvel

O aplicativo pode ser acessado diretamente pelo navegador do smartphone.

---

# 📂 Estrutura do Projeto

```text
fissuramap/
│
├── dataset/
│   ├── images/
│   └── labels/
│
├── modelos/
│   └── best.pt
│
├── organizar_dataset.py
├── treinar_modelo.py
├── diagnostico_fissura.py
├── app.py
├── requirements.txt
└── README.md
```

---

# 🧠 Arquitetura da Solução

```text
Imagem
    ↓
YOLOv8 Segmentation
    ↓
Máscara da fissura
    ↓
Extração de informações
    ↓
Posição
Orientação
Área afetada
Nível de atenção
    ↓
Aplicação Streamlit
```

---

# 📊 Dataset

O conjunto de dados utilizado contém aproximadamente:

* 1551 imagens
* Labels no formato YOLO Segmentation

Formato dos labels:

```text
classe x1 y1 x2 y2 x3 y3 ...
```

Cada label representa um polígono delimitando a região da fissura.

---

# ⚙️ Preparação do Dataset

Estrutura esperada:

```text
dataset/
├── images/
└── labels/
```

Executar:

```bash
python organizar_dataset.py
```

O script divide automaticamente os dados em:

```text
70% treino
20% validação
10% teste
```

e gera:

```text
dataset_yolo/
├── images/
├── labels/
└── data.yaml
```

---

# 🏋️ Treinamento

Executar:

```bash
python treinar_modelo.py
```

Configurações utilizadas:

| Parâmetro  | Valor       |
| ---------- | ----------- |
| Modelo     | YOLOv8n-seg |
| Epochs     | 60          |
| Batch Size | 16          |
| Image Size | 640         |
| Patience   | 15          |

---

# 📈 Resultados Obtidos

## Detecção (Bounding Boxes)

| Métrica   | Valor  |
| --------- | ------ |
| Precision | 84,58% |
| Recall    | 71,66% |
| mAP50     | 79,77% |
| mAP50-95  | 60,43% |

## Segmentação

| Métrica   | Valor  |
| --------- | ------ |
| Precision | 77,65% |
| Recall    | 63,82% |
| mAP50     | 64,36% |
| mAP50-95  | 25,55% |

---

# 📱 Execução da Aplicação

Instalar dependências:

```bash
pip install -r requirements.txt
```

Executar:

```bash
streamlit run app.py
```

O navegador abrirá automaticamente:

```text
http://localhost:8501
```

---

# 📸 Como Utilizar

1. Abra o aplicativo.
2. Escolha:

   * Selecionar imagem da galeria;
   * Tirar foto utilizando a câmera.
3. Aguarde o processamento.
4. Visualize:

   * fissuras detectadas;
   * posição;
   * orientação;
   * nível de atenção;
   * máscara segmentada.

---

# 🔧 Decisões de Engenharia

## Por que YOLOv8 Segmentation?

Os labels fornecidos já continham máscaras poligonais.

Dessa forma, a segmentação aproveita melhor as informações disponíveis do que uma simples detecção por caixas delimitadoras.

## Por que YOLOv8n?

O objetivo do desafio prevê utilização em dispositivos móveis.

O modelo YOLOv8n apresenta:

* menor consumo de memória;
* menor tempo de inferência;
* maior viabilidade para smartphones.

## Por que Streamlit?

Permite:

* execução em navegador;
* compatibilidade com celulares;
* rápida prototipação;
* facilidade de demonstração.

## Por que Google Colab?

O treinamento inicial foi realizado localmente.

Entretanto, devido ao tamanho do conjunto de dados e ao custo computacional do treinamento de redes neurais convolucionais, optou-se pela utilização do Google Colab com aceleração por GPU, reduzindo significativamente o tempo de treinamento.

---

# 🔮 Trabalhos Futuros

* Exportação para TensorFlow Lite;
* Execução totalmente offline em smartphones;
* Classificação automática de severidade;
* Histórico de inspeções;
* Georreferenciamento das ocorrências.

---

# 👩‍💻 Autora

Fernanda S. A. Vale

Bootcamp de Ciência de Dados e Inteligência Artificial
