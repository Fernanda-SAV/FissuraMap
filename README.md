# FissuraMap - Desafio 2

Sistema web em Streamlit para localização de fissuras e trincas em paredes usando YOLOv8-seg.

## Estrutura esperada

Coloque o dataset fornecido dentro da pasta:

```text
fissuramap_desafio2/
├── dataset/
│   ├── images/
│   └── labels/
├── modelos/
├── organizar_dataset.py
├── treinar_modelo.py
├── diagnostico_fissura.py
├── app.py
├── visualizar_label.py
└── requirements.txt
```

## Rodar no Google Colab

1. Suba a pasta `fissuramap_desafio2` para o Google Drive.
2. No Colab, ative GPU: `Ambiente de execução > Alterar tipo de ambiente de execução > GPU`.
3. Execute:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/fissuramap_desafio2
```

```python
!pip install -r requirements.txt
```

```python
!python organizar_dataset.py
```

```python
!python treinar_modelo.py
```

## Rodar o app Streamlit no Colab

```python
!npm install -g localtunnel
```

```python
!streamlit run app.py --server.port 8501 & npx localtunnel --port 8501
```

O Colab vai gerar um link público temporário. Abra o link no navegador ou no celular.

## Saída do modelo

Após o treinamento, o melhor modelo será salvo em:

```text
modelos/best.pt
```

Esse arquivo é usado automaticamente pelo `app.py`.
