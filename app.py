from pathlib import Path
import tempfile
import cv2
import streamlit as st
from diagnostico_fissura import FissuraMap

#nesse código eu configurei a página do Streamlit, 
#criei a interface para o usuário enviar uma 
#imagem da parede, e exibir os resultados da análise 
#de fissuras, incluindo métricas e imagens marcadas. 
#O código também lida com erros caso o modelo de detecção 
#de fissuras não esteja disponível.



st.set_page_config(
    page_title="FissuraMap",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🧱 FissuraMap")
st.caption("Sistema de inspeção visual para localização de fissuras e trincas em paredes")

st.info(
    """
    **Como usar:**

    1. Envie uma imagem da galeria ou tire uma foto com a câmera do dispositivo.
    2. O sistema marca as fissuras encontradas na parede.
    3. A aplicação informa posição, confiança, área afetada e nível de atenção.
    """
)

col_entrada, col_config = st.columns([2, 1])

with col_entrada:
    modo_entrada = st.radio(
        "Escolha a forma de entrada da imagem:",
        ["Selecionar imagem da galeria", "Tirar foto agora"],
        horizontal=True,
    )

with col_config:
    confianca = st.slider(
        "Confiança mínima da detecção",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

arquivo = None

if modo_entrada == "Selecionar imagem da galeria":
    arquivo = st.file_uploader(
        "Envie uma imagem da parede",
        type=["jpg", "jpeg", "png"],
    )
else:
    arquivo = st.camera_input("Tire uma foto da parede")

if arquivo is None:
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
    temp.write(arquivo.getvalue())
    caminho_temp = Path(temp.name)

try:
    analisador = FissuraMap(confianca_minima=confianca)
    dados = analisador.analisar(caminho_temp)
except FileNotFoundError as erro:
    st.error(str(erro))
    st.warning("Treine o modelo primeiro. O arquivo esperado é: modelos/best.pt")
    st.stop()

st.divider()
st.subheader("Resultado da inspeção")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Fissuras localizadas", dados["quantidade"])
m2.metric("Confiança média", f"{dados['confianca_media']}%")
m3.metric("Área afetada", f"{dados['area_total_percentual']}%")
m4.metric("Nível de atenção", dados["nivel_atencao"].capitalize())

if dados["nivel_atencao"] in ["alto", "crítico"]:
    st.error(dados["mensagem"])
elif dados["quantidade"] > 0:
    st.warning(dados["mensagem"])
else:
    st.success(dados["mensagem"])

aba1, aba2, aba3 = st.tabs(["Imagem marcada", "Máscara das fissuras", "Detalhes por região"])

with aba1:
    imagem_rgb = cv2.cvtColor(dados["imagem_marcada"], cv2.COLOR_BGR2RGB)
    st.image(imagem_rgb, caption="Imagem com fissuras destacadas", use_container_width=True)

with aba2:
    st.image(dados["mascara_total"], caption="Máscara total das regiões detectadas", use_container_width=True)

with aba3:
    if not dados["fissuras"]:
        st.write("Nenhuma fissura detalhada para exibir.")
    else:
        for fissura in dados["fissuras"]:
            st.markdown(f"### Fissura {fissura['id']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confiança", f"{fissura['confianca']}%")
            c2.metric("Área", f"{fissura['area_percentual']}%")
            c3.metric("Extensão relativa", f"{fissura['extensao_relativa']}%")
            c4.write(f"**Posição:** {fissura['posicao']}")
            st.write(f"**Orientação:** {fissura['orientacao']}")
            st.divider()
