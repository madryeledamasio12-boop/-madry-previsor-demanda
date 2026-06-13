# ============================================================
# PREVISOR DE DEMANDA SEMANAL – MADRY PAPELARIA CRIATIVA
# Desenvolvido com Streamlit, Pandas, NumPy, Plotly e Scikit-Learn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ReportLab para geração de PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Previsor de Demanda – Madry Papelaria",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PERSONALIZADO
# ============================================================
st.markdown("""
<style>
    /* Fundo principal */
    .main { background-color: #F9F7F4; }

    /* Título principal */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2C3E50;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #7F8C8D;
        margin-bottom: 1.5rem;
    }

    /* Cards de métricas */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #E67E22;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #7F8C8D;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #2C3E50;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #95A5A6;
    }

    /* Alerta de recomendação */
    .rec-box {
        background: #FEF9F0;
        border: 1px solid #F39C12;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .rec-title {
        font-size: 1rem;
        font-weight: 700;
        color: #D35400;
        margin-bottom: 0.5rem;
    }
    .rec-text {
        font-size: 0.92rem;
        color: #444;
        line-height: 1.6;
    }

    /* Badges */
    .badge-green  { background:#27AE60; color:white; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }
    .badge-yellow { background:#F39C12; color:white; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }
    .badge-red    { background:#E74C3C; color:white; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }

    /* Seção */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2C3E50;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #E67E22;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #2C3E50;
    }
    section[data-testid="stSidebar"] * {
        color: #ECF0F1 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #BDC3C7 !important;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #F39C12 !important;
    }

    /* Tabela de ranking */
    .ranking-table { width: 100%; border-collapse: collapse; }
    .ranking-table th { background: #2C3E50; color: white; padding: 8px 12px; text-align: left; font-size: 0.82rem; }
    .ranking-table td { padding: 7px 12px; border-bottom: 1px solid #EEE; font-size: 0.85rem; }
    .ranking-table tr:hover td { background: #FEF9F0; }

    /* Aviso */
    .warning-box {
        background: #FDF2E9;
        border-left: 4px solid #E67E22;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #784212;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNÇÕES: LEITURA E VALIDAÇÃO DE DADOS
# ============================================================

def parse_manual_input(text: str):
    """
    Converte texto digitado manualmente em lista de floats.
    Aceita vírgulas ou ponto-e-vírgula como separadores.
    """
    text = text.strip().replace(";", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    valores = []
    for p in parts:
        try:
            v = float(p.replace(",", "."))
            valores.append(v)
        except ValueError:
            return None, f"Valor inválido encontrado: '{p}'. Use apenas números."
    return valores, None


def validate_data(valores):
    """
    Valida os dados de entrada conforme regras de negócio.
    Retorna (ok, mensagem_erro).
    """
    if not valores or len(valores) == 0:
        return False, "Nenhum dado informado."
    if len(valores) < 8:
        return False, f"São necessárias pelo menos 8 semanas de histórico. Você inseriu {len(valores)}."
    if any(v < 0 for v in valores):
        return False, "Valores negativos não são permitidos."
    if any(v > 1_000_000 for v in valores):
        return False, "Valores acima de 1.000.000 parecem incorretos. Verifique os dados."
    return True, None


def load_file(uploaded_file):
    """
    Carrega dados de CSV ou XLSX.
    Espera colunas: Semana | Demanda (ou apenas uma coluna numérica).
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        # Tenta encontrar coluna de demanda
        col_demanda = None
        for col in df.columns:
            if col.strip().lower() in ["demanda", "vendas", "demand", "sales", "quantidade"]:
                col_demanda = col
                break

        if col_demanda is None:
            # Assume segunda coluna ou única coluna numérica
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                return None, "Nenhuma coluna numérica encontrada no arquivo."
            # Prefere a última coluna numérica (geralmente Demanda)
            col_demanda = num_cols[-1]

        valores = df[col_demanda].dropna().tolist()
        return valores, None
    except Exception as e:
        return None, f"Erro ao ler arquivo: {str(e)}"


# ============================================================
# FUNÇÕES: MÉTODOS DE PREVISÃO
# ============================================================

def metodo_ingenuo(dados, horizonte):
    """
    Método Ingênuo: repete o último valor observado para todos os períodos futuros.
    """
    ultimo = dados[-1]
    previsoes = [ultimo] * horizonte
    # Histórico "previsto" (lag 1)
    hist_prev = [None] + list(dados[:-1])
    return previsoes, hist_prev


def media_movel_simples(dados, janela, horizonte):
    """
    Média Móvel Simples com janela configurável.
    Previsão futura = média das últimas `janela` semanas.
    """
    if len(dados) < janela:
        return None, None

    hist_prev = [None] * janela
    for i in range(janela, len(dados)):
        hist_prev.append(np.mean(dados[i - janela:i]))

    ultimo_prev = np.mean(dados[-janela:])
    previsoes = [ultimo_prev] * horizonte
    return previsoes, hist_prev


def media_movel_ponderada(dados, pesos, horizonte):
    """
    Média Móvel Ponderada com pesos configuráveis.
    Os pesos são aplicados da mais antiga para a mais recente observação.
    """
    n = len(pesos)
    if len(dados) < n:
        return None, None

    pesos_norm = np.array(pesos) / np.sum(pesos)

    hist_prev = [None] * n
    for i in range(n, len(dados)):
        janela = dados[i - n:i]
        hist_prev.append(np.dot(janela, pesos_norm))

    ultimo_prev = np.dot(dados[-n:], pesos_norm)
    previsoes = [ultimo_prev] * horizonte
    return previsoes, hist_prev


def suavizacao_exponencial(dados, alfa, horizonte):
    """
    Suavização Exponencial Simples.
    F(t+1) = alfa * D(t) + (1 - alfa) * F(t)
    Inicializa com a primeira observação.
    """
    F = [dados[0]]
    for t in range(1, len(dados)):
        F.append(alfa * dados[t - 1] + (1 - alfa) * F[t - 1])

    # Previsão futura iterativa
    F_futuro = F[-1]
    previsoes = []
    for _ in range(horizonte):
        F_futuro = alfa * dados[-1] + (1 - alfa) * F_futuro
        previsoes.append(F_futuro)

    return previsoes, F


def regressao_linear(dados, horizonte):
    """
    Regressão Linear Simples via Scikit-Learn.
    X = semana (índice), y = demanda.
    """
    n = len(dados)
    X = np.arange(1, n + 1).reshape(-1, 1)
    y = np.array(dados)

    modelo = LinearRegression()
    modelo.fit(X, y)

    # Histórico ajustado
    hist_prev = modelo.predict(X).tolist()

    # Previsões futuras
    X_futuro = np.arange(n + 1, n + horizonte + 1).reshape(-1, 1)
    previsoes = modelo.predict(X_futuro).tolist()

    return previsoes, hist_prev, modelo.coef_[0], modelo.intercept_


# ============================================================
# FUNÇÕES: INDICADORES DE ERRO
# ============================================================

def calcular_erros(real, previsto):
    """
    Calcula MAE, MAPE e RMSE entre valores reais e previstos.
    Ignora posições onde previsto é None.
    """
    pares = [(r, p) for r, p in zip(real, previsto) if p is not None]
    if not pares:
        return None, None, None

    reais = np.array([p[0] for p in pares])
    prev  = np.array([p[1] for p in pares])

    mae  = np.mean(np.abs(reais - prev))
    rmse = np.sqrt(np.mean((reais - prev) ** 2))

    # MAPE: evita divisão por zero
    nonzero = reais != 0
    if nonzero.sum() == 0:
        mape = None
    else:
        mape = np.mean(np.abs((reais[nonzero] - prev[nonzero]) / reais[nonzero])) * 100

    return round(mae, 2), round(mape, 2) if mape is not None else None, round(rmse, 2)


# ============================================================
# FUNÇÕES: ANÁLISE DA DEMANDA
# ============================================================

def analisar_tendencia(dados):
    """
    Analisa tendência e variabilidade dos dados históricos.
    Retorna: tipo (crescimento/queda/estabilidade/alta_variabilidade), slope, cv
    """
    n = len(dados)
    X = np.arange(n).reshape(-1, 1)
    y = np.array(dados)
    modelo = LinearRegression().fit(X, y)
    slope = modelo.coef_[0]

    media = np.mean(dados)
    std   = np.std(dados)
    cv    = (std / media) * 100 if media != 0 else 0

    if cv > 25:
        tipo = "alta_variabilidade"
    elif slope > 1.5:
        tipo = "crescimento"
    elif slope < -1.5:
        tipo = "queda"
    else:
        tipo = "estabilidade"

    return tipo, round(slope, 3), round(cv, 2)


def calcular_estoque_seguranca(dados, nivel_servico=1.65):
    """
    Estoque de Segurança = Z * Desvio Padrão da Demanda.
    Padrão: nível de serviço 95% → Z = 1,65.
    """
    std = np.std(dados, ddof=1)
    es  = nivel_servico * std
    return round(es, 1), round(std, 2)


# ============================================================
# FUNÇÕES: RECOMENDAÇÃO GERENCIAL
# ============================================================

def gerar_recomendacao(tipo_tendencia, slope, cv, produto, cap_prod, demanda_prevista):
    """
    Gera texto de recomendação gerencial baseado na análise da demanda.
    """
    rec = {
        "crescimento": (
            f"📈 A demanda de <b>{produto}</b> apresenta tendência de <b>crescimento</b> "
            f"(+{slope:.1f} un/semana). Recomenda-se aumentar gradualmente a produção, "
            "revisar o estoque de matérias-primas (capas, espirais e folhas), além de avaliar "
            "a necessidade de ampliar a capacidade produtiva para atender à demanda crescente. "
            "Considere negociar volumes maiores com fornecedores para reduzir custos unitários."
        ),
        "queda": (
            f"📉 A demanda de <b>{produto}</b> apresenta tendência de <b>redução</b> "
            f"({slope:.1f} un/semana). Recomenda-se <b>cautela na produção</b> para evitar "
            "excesso de estoque e custos desnecessários. Avalie ações promocionais para "
            "estimular vendas e revise as metas de produção semanal. Reduza pedidos de "
            "matéria-prima até estabilização da demanda."
        ),
        "estabilidade": (
            f"⚖️ A demanda de <b>{produto}</b> apresenta comportamento <b>estável</b>, "
            "permitindo planejamento produtivo com menor risco. Mantenha o nível atual de "
            "produção, garanta o estoque de segurança calculado e aproveite a previsibilidade "
            "para otimizar compras de matéria-prima com melhores condições de prazo e custo."
        ),
        "alta_variabilidade": (
            f"⚠️ A demanda de <b>{produto}</b> apresenta <b>oscilações significativas</b> "
            f"(CV = {cv:.1f}%). Recomenda-se analisar sazonalidade, campanhas promocionais "
            "e períodos de volta às aulas. Mantenha um estoque de segurança robusto e considere "
            "produção sob demanda (make-to-order) para itens personalizados, reduzindo o risco "
            "de obsolescência de estoque."
        ),
    }

    texto_base = rec.get(tipo_tendencia, "Analise os dados com atenção.")

    # Alerta de capacidade
    if cap_prod > 0 and demanda_prevista:
        dem_max = max(demanda_prevista)
        if dem_max > cap_prod:
            texto_base += (
                f" <br><br>🔴 <b>Alerta Crítico:</b> A demanda prevista máxima "
                f"({dem_max:.0f} un) <b>supera</b> a capacidade produtiva ({cap_prod} un/sem). "
                "Avalie horas extras, terceirização ou expansão de capacidade imediatamente."
            )
        elif dem_max > cap_prod * 0.85:
            texto_base += (
                f" <br><br>🟡 <b>Atenção:</b> A demanda prevista ({dem_max:.0f} un) está "
                f"próxima da capacidade ({cap_prod} un/sem). Monitore de perto e prepare "
                "contingências produtivas."
            )

    return texto_base


# ============================================================
# FUNÇÕES: GRÁFICOS PLOTLY
# ============================================================

COR_HISTORICO  = "#2C3E50"
COR_PREV       = "#E67E22"
COR_TENDENCIA  = "#8E44AD"
CORES_METODOS  = ["#3498DB", "#E67E22", "#2ECC71", "#9B59B6", "#E74C3C"]

def grafico_historico(dados, semanas):
    """Gráfico 1: Série histórica de vendas."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=semanas, y=dados,
        mode="lines+markers",
        name="Vendas Reais",
        line=dict(color=COR_HISTORICO, width=2.5),
        marker=dict(size=7, color=COR_HISTORICO),
        fill="tozeroy",
        fillcolor="rgba(44,62,80,0.07)"
    ))
    # Linha de média
    media = np.mean(dados)
    fig.add_hline(y=media, line_dash="dot", line_color="#95A5A6",
                  annotation_text=f"Média: {media:.0f}", annotation_position="right")
    fig.update_layout(
        title="📊 Histórico de Vendas Semanais",
        xaxis_title="Semana", yaxis_title="Unidades Vendidas",
        template="plotly_white", height=380,
        legend=dict(orientation="h", y=-0.15)
    )
    return fig


def grafico_historico_previsao(dados, semanas, previsoes, nome_metodo, horizonte):
    """Gráfico 2: Histórico + Previsão futura."""
    sem_futuras = list(range(semanas[-1] + 1, semanas[-1] + horizonte + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=semanas, y=dados,
        mode="lines+markers", name="Histórico Real",
        line=dict(color=COR_HISTORICO, width=2.5),
        marker=dict(size=7)
    ))
    # Ponto de ligação entre histórico e previsão
    x_link = [semanas[-1]] + sem_futuras
    y_link = [dados[-1]] + previsoes
    fig.add_trace(go.Scatter(
        x=x_link, y=y_link,
        mode="lines+markers", name=f"Previsão ({nome_metodo})",
        line=dict(color=COR_PREV, width=2.5, dash="dash"),
        marker=dict(size=8, symbol="diamond", color=COR_PREV)
    ))
    # Faixa de incerteza (±10%)
    y_upper = [v * 1.10 for v in previsoes]
    y_lower = [max(0, v * 0.90) for v in previsoes]
    fig.add_trace(go.Scatter(
        x=sem_futuras + sem_futuras[::-1],
        y=y_upper + y_lower[::-1],
        fill="toself", fillcolor="rgba(230,126,34,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Faixa ±10%", showlegend=True
    ))
    fig.update_layout(
        title=f"🔮 Previsão de Demanda – {nome_metodo}",
        xaxis_title="Semana", yaxis_title="Unidades",
        template="plotly_white", height=400,
        legend=dict(orientation="h", y=-0.18)
    )
    return fig


def grafico_comparacao_metodos(dados, semanas, resultados_metodos):
    """Gráfico 3: Comparação visual dos métodos no período histórico."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=semanas, y=dados,
        mode="lines+markers", name="Real",
        line=dict(color=COR_HISTORICO, width=3),
        marker=dict(size=8)
    ))
    for i, (nome, _, hist_prev, _) in enumerate(resultados_metodos):
        y_vals = [v if v is not None else np.nan for v in hist_prev]
        fig.add_trace(go.Scatter(
            x=semanas, y=y_vals,
            mode="lines", name=nome,
            line=dict(color=CORES_METODOS[i % len(CORES_METODOS)], width=1.8, dash="dot")
        ))
    fig.update_layout(
        title="📐 Comparação dos Métodos de Previsão",
        xaxis_title="Semana", yaxis_title="Unidades",
        template="plotly_white", height=420,
        legend=dict(orientation="h", y=-0.22)
    )
    return fig


def grafico_tendencia(dados, semanas):
    """Gráfico 4: Tendência linear da demanda."""
    X = np.arange(len(dados)).reshape(-1, 1)
    y = np.array(dados)
    modelo = LinearRegression().fit(X, y)
    tendencia = modelo.predict(X).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=semanas, y=dados,
        name="Demanda Real",
        marker_color="rgba(44,62,80,0.5)"
    ))
    fig.add_trace(go.Scatter(
        x=semanas, y=tendencia,
        mode="lines", name="Tendência Linear",
        line=dict(color=COR_TENDENCIA, width=3)
    ))
    fig.update_layout(
        title="📈 Tendência da Demanda",
        xaxis_title="Semana", yaxis_title="Unidades",
        template="plotly_white", height=380,
        legend=dict(orientation="h", y=-0.18)
    )
    return fig


# ============================================================
# FUNÇÃO: GERAÇÃO DO PDF COM REPORTLAB
# ============================================================

def gerar_pdf(
    produto, dados, semanas, metodo_nome, previsoes,
    mae, mape, rmse, tipo_tendencia, slope, cv,
    es, std_demanda, cap_prod, recomendacao_texto
):
    """
    Gera relatório PDF completo usando ReportLab.
    Retorna bytes do PDF.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm
    )

    estilos = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle(
        "Titulo", parent=estilos["Title"],
        fontSize=18, textColor=colors.HexColor("#2C3E50"),
        spaceAfter=6, alignment=TA_CENTER
    )
    estilo_subtitulo = ParagraphStyle(
        "SubTitulo", parent=estilos["Normal"],
        fontSize=11, textColor=colors.HexColor("#7F8C8D"),
        spaceAfter=18, alignment=TA_CENTER
    )
    estilo_secao = ParagraphStyle(
        "Secao", parent=estilos["Heading2"],
        fontSize=12, textColor=colors.HexColor("#E67E22"),
        spaceAfter=6, spaceBefore=14,
        borderPad=4
    )
    estilo_body = ParagraphStyle(
        "Body", parent=estilos["Normal"],
        fontSize=10, textColor=colors.HexColor("#333"),
        spaceAfter=6, leading=14, alignment=TA_JUSTIFY
    )
    estilo_rodape = ParagraphStyle(
        "Rodape", parent=estilos["Normal"],
        fontSize=8, textColor=colors.grey,
        alignment=TA_CENTER
    )

    conteudo = []

    # Cabeçalho
    conteudo.append(Paragraph("📚 Madry Papelaria Criativa", estilo_titulo))
    conteudo.append(Paragraph("Relatório de Previsão de Demanda Semanal", estilo_subtitulo))
    conteudo.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#E67E22")))
    conteudo.append(Spacer(1, 0.4*cm))

    # Informações básicas
    conteudo.append(Paragraph("1. Informações do Projeto", estilo_secao))
    info_data = [
        ["Produto Analisado:", produto],
        ["Semanas de Histórico:", str(len(dados))],
        ["Método de Previsão:", metodo_nome],
        ["Horizonte Previsto:", f"{len(previsoes)} semana(s)"],
        ["Capacidade Produtiva:", f"{cap_prod} unidades/semana" if cap_prod > 0 else "Não informada"],
    ]
    t_info = Table(info_data, colWidths=[5.5*cm, 10*cm])
    t_info.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#2C3E50")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#FEF9F0"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDD")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    conteudo.append(t_info)
    conteudo.append(Spacer(1, 0.4*cm))

    # Histórico de vendas
    conteudo.append(Paragraph("2. Histórico de Vendas (Unidades/Semana)", estilo_secao))
    hist_cabecalho = ["Semana", "Demanda"]
    hist_rows = [hist_cabecalho] + [[str(s), str(int(d))] for s, d in zip(semanas, dados)]
    t_hist = Table(hist_rows, colWidths=[4*cm, 4*cm])
    t_hist.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9F7F4")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDD")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    conteudo.append(t_hist)
    conteudo.append(Spacer(1, 0.4*cm))

    # Indicadores de erro
    conteudo.append(Paragraph("3. Indicadores de Erro do Método Selecionado", estilo_secao))
    erro_rows = [
        ["Indicador", "Valor", "Interpretação"],
        ["MAE (Erro Médio Absoluto)", f"{mae:.2f} un" if mae else "N/A", "Quanto o modelo erra em média"],
        ["MAPE (Erro % Médio)", f"{mape:.2f}%" if mape else "N/A", "Erro relativo à demanda real"],
        ["RMSE (Raiz do Erro Quadrático)", f"{rmse:.2f} un" if rmse else "N/A", "Penaliza erros grandes"],
    ]
    t_erro = Table(erro_rows, colWidths=[5*cm, 3.5*cm, 7*cm])
    t_erro.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E67E22")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FEF9F0")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDD")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ]))
    conteudo.append(t_erro)
    conteudo.append(Spacer(1, 0.4*cm))

    # Previsões futuras
    conteudo.append(Paragraph("4. Previsões de Demanda Futura", estilo_secao))
    prev_cabecalho = ["Semana Futura", "Demanda Prevista (un)"]
    prev_rows = [prev_cabecalho] + [
        [f"Sem. {semanas[-1] + i + 1}", f"{v:.0f}"]
        for i, v in enumerate(previsoes)
    ]
    t_prev = Table(prev_rows, colWidths=[5*cm, 5*cm])
    t_prev.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9F7F4")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDD")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    conteudo.append(t_prev)
    conteudo.append(Spacer(1, 0.4*cm))

    # Estoque de segurança
    conteudo.append(Paragraph("5. Gestão de Estoque de Segurança", estilo_secao))
    es_texto = (
        f"Desvio Padrão da Demanda Histórica: {std_demanda:.2f} unidades\n"
        f"Estoque de Segurança Recomendado (Z=1,65 / 95%): {es:.0f} unidades\n\n"
        "O estoque de segurança protege a empresa contra variações inesperadas na demanda "
        "e nos prazos de reposição de matéria-prima. Manter esse nível reduz o risco de "
        "ruptura de estoque e perda de vendas."
    )
    conteudo.append(Paragraph(es_texto, estilo_body))
    conteudo.append(Spacer(1, 0.3*cm))

    # Análise da tendência
    conteudo.append(Paragraph("6. Análise da Tendência de Demanda", estilo_secao))
    tipo_label = {
        "crescimento": "📈 Crescimento",
        "queda": "📉 Queda",
        "estabilidade": "⚖️ Estabilidade",
        "alta_variabilidade": "⚠️ Alta Variabilidade"
    }
    conteudo.append(Paragraph(
        f"Tendência identificada: <b>{tipo_label.get(tipo_tendencia, tipo_tendencia)}</b> | "
        f"Slope: {slope:.3f} un/semana | Coeficiente de Variação: {cv:.2f}%",
        estilo_body
    ))

    # Recomendação gerencial (sem HTML)
    conteudo.append(Paragraph("7. Recomendação Gerencial", estilo_secao))
    # Remove tags HTML simples para o PDF
    rec_limpa = recomendacao_texto.replace("<b>", "").replace("</b>", "")
    rec_limpa = rec_limpa.replace("<br><br>", "\n\n").replace("<br>", "\n")
    for emoji in ["📈", "📉", "⚖️", "⚠️", "🔴", "🟡"]:
        rec_limpa = rec_limpa.replace(emoji, "")
    conteudo.append(Paragraph(rec_limpa.strip(), estilo_body))
    conteudo.append(Spacer(1, 0.5*cm))

    # Rodapé
    conteudo.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDD")))
    conteudo.append(Spacer(1, 0.2*cm))
    conteudo.append(Paragraph(
        "Madry Papelaria Criativa – Sistema de Previsão de Demanda | Gerado automaticamente",
        estilo_rodape
    ))

    doc.build(conteudo)
    buffer.seek(0)
    return buffer.read()


# ============================================================
# INTERFACE PRINCIPAL – SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📚 Madry Papelaria")
    st.markdown("---")

    produto = st.text_input("🏷️ Produto", value="Caderno Universitário Personalizado")

    horizonte_opcoes = {
        "1 semana": 1, "2 semanas": 2, "4 semanas": 4,
        "8 semanas": 8, "12 semanas": 12
    }
    horizonte_label = st.selectbox("📅 Horizonte de Previsão", list(horizonte_opcoes.keys()), index=2)
    horizonte = horizonte_opcoes[horizonte_label]

    cap_prod = st.number_input(
        "🏭 Capacidade Produtiva (un/sem)", min_value=0, value=250, step=10
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parâmetros dos Métodos")

    alfa = st.slider("Alfa (Suavização Exponencial)", 0.1, 0.9, 0.3, 0.05)

    pesos_str = st.text_input(
        "Pesos Média Ponderada (mais antiga → mais recente)",
        value="0.2,0.3,0.5",
        help="Informe pesos separados por vírgula. Ex: 0.2,0.3,0.5 para 3 semanas."
    )

    # Parse dos pesos
    try:
        pesos = [float(p.strip()) for p in pesos_str.split(",") if p.strip()]
        if len(pesos) < 2:
            pesos = [0.2, 0.3, 0.5]
    except ValueError:
        pesos = [0.2, 0.3, 0.5]

    st.markdown("---")
    st.markdown("### 📌 Método Principal")
    metodo_selecionado = st.selectbox(
        "Para previsão futura e PDF",
        ["Método Ingênuo", "Média Móvel (3)", "Média Móvel (4)", "Média Móvel (5)",
         "Média Ponderada", "Suavização Exponencial", "Regressão Linear"]
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#BDC3C7'>Madry Papelaria Criativa<br>"
        "Sistema de PCP – Previsão de Demanda</small>",
        unsafe_allow_html=True
    )


# ============================================================
# ÁREA PRINCIPAL
# ============================================================

st.markdown('<div class="main-title">📚 Previsor de Demanda Semanal Inteligente</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Planejamento e Controle da Produção para Papelarias Personalizadas</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# ENTRADA DE DADOS
# ------------------------------------------------------------
st.markdown('<div class="section-header">📥 Entrada de Dados Históricos</div>', unsafe_allow_html=True)

tab_manual, tab_upload = st.tabs(["✏️ Digitação Manual", "📂 Upload de Arquivo"])

dados_raw = None
fonte_dados = ""

with tab_manual:
    st.markdown("Digite as vendas semanais separadas por vírgula (mínimo 8 semanas):")
    texto_dados = st.text_area(
        "Histórico de Vendas",
        value="120,125,130,128,140,150,155,148,160,165,170,168",
        height=80,
        label_visibility="collapsed"
    )
    if st.button("▶ Processar Dados Digitados", type="primary", use_container_width=True):
        dados_raw, erro = parse_manual_input(texto_dados)
        if erro:
            st.error(f"❌ {erro}")
            dados_raw = None
        else:
            st.session_state["dados"] = dados_raw
            st.session_state["fonte"] = "manual"
            st.success(f"✅ {len(dados_raw)} semanas carregadas com sucesso!")

with tab_upload:
    st.markdown("Envie um arquivo CSV ou XLSX com colunas **Semana** e **Demanda**:")
    arquivo = st.file_uploader("Selecionar arquivo", type=["csv", "xlsx"], label_visibility="collapsed")
    if arquivo:
        dados_raw, erro = load_file(arquivo)
        if erro:
            st.error(f"❌ {erro}")
            dados_raw = None
        else:
            st.session_state["dados"] = dados_raw
            st.session_state["fonte"] = arquivo.name
            st.success(f"✅ {len(dados_raw)} semanas carregadas de '{arquivo.name}'!")

# Recupera dados da sessão
if "dados" in st.session_state:
    dados_raw = st.session_state["dados"]

# ------------------------------------------------------------
# PROCESSAMENTO PRINCIPAL
# ------------------------------------------------------------
if dados_raw:
    ok, msg_erro = validate_data(dados_raw)

    if not ok:
        st.error(f"❌ {msg_erro}")
    else:
        dados   = [float(v) for v in dados_raw]
        n       = len(dados)
        semanas = list(range(1, n + 1))

        # --------------------------------------------------------
        # KPIs DO HISTÓRICO
        # --------------------------------------------------------
        st.markdown('<div class="section-header">📊 Resumo do Histórico</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Semanas Analisadas</div>
                <div class="metric-value">{n}</div>
                <div class="metric-sub">semanas de histórico</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Demanda Média</div>
                <div class="metric-value">{np.mean(dados):.0f}</div>
                <div class="metric-sub">unidades/semana</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Demanda Máxima</div>
                <div class="metric-value">{max(dados):.0f}</div>
                <div class="metric-sub">unidades/semana</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Demanda Mínima</div>
                <div class="metric-value">{min(dados):.0f}</div>
                <div class="metric-sub">unidades/semana</div>
            </div>""", unsafe_allow_html=True)

        # --------------------------------------------------------
        # CALCULAR TODOS OS MÉTODOS
        # --------------------------------------------------------
        resultados_metodos = []  # (nome, previsoes, hist_prev, erros)

        # 1. Ingênuo
        prev_ing, hist_ing = metodo_ingenuo(dados, horizonte)
        mae_i, mape_i, rmse_i = calcular_erros(dados, hist_ing)
        resultados_metodos.append(("Método Ingênuo", prev_ing, hist_ing, (mae_i, mape_i, rmse_i)))

        # 2–4. Médias Móveis
        for j in [3, 4, 5]:
            if len(dados) >= j:
                prev_mm, hist_mm = media_movel_simples(dados, j, horizonte)
                if prev_mm:
                    mae_m, mape_m, rmse_m = calcular_erros(dados, hist_mm)
                    resultados_metodos.append((f"Média Móvel ({j})", prev_mm, hist_mm, (mae_m, mape_m, rmse_m)))

        # 5. Ponderada
        if len(dados) >= len(pesos):
            prev_mp, hist_mp = media_movel_ponderada(dados, pesos, horizonte)
            if prev_mp:
                mae_p, mape_p, rmse_p = calcular_erros(dados, hist_mp)
                resultados_metodos.append(("Média Ponderada", prev_mp, hist_mp, (mae_p, mape_p, rmse_p)))

        # 6. Suavização Exponencial
        prev_se, hist_se = suavizacao_exponencial(dados, alfa, horizonte)
        mae_s, mape_s, rmse_s = calcular_erros(dados, hist_se)
        resultados_metodos.append((f"Suav. Exp. (α={alfa})", prev_se, hist_se, (mae_s, mape_s, rmse_s)))

        # 7. Regressão Linear
        prev_rl, hist_rl, coef_rl, inter_rl = regressao_linear(dados, horizonte)
        mae_r, mape_r, rmse_r = calcular_erros(dados, hist_rl)
        resultados_metodos.append(("Regressão Linear", prev_rl, hist_rl, (mae_r, mape_r, rmse_r)))

        # Ranking por MAE
        rank = sorted(
            [(n, e[0]) for n, _, _, e in resultados_metodos if e[0] is not None],
            key=lambda x: x[1]
        )
        melhor_metodo_nome = rank[0][0] if rank else "N/A"

        # --------------------------------------------------------
        # MÉTODO SELECIONADO
        # --------------------------------------------------------
        mapa_metodo = {
            "Método Ingênuo": 0,
            "Média Móvel (3)": 1,
            "Média Móvel (4)": 2,
            "Média Móvel (5)": 3,
            "Média Ponderada": 4,
            "Suavização Exponencial": 5,
            "Regressão Linear": 6,
        }
        idx_sel = min(mapa_metodo.get(metodo_selecionado, 0), len(resultados_metodos) - 1)
        nome_sel, previsoes_sel, hist_sel, erros_sel = resultados_metodos[idx_sel]
        mae_sel, mape_sel, rmse_sel = erros_sel

        # --------------------------------------------------------
        # GRÁFICOS
        # --------------------------------------------------------
        st.markdown('<div class="section-header">📉 Gráficos e Análise Visual</div>', unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(grafico_historico(dados, semanas), use_container_width=True)
        with col_g2:
            st.plotly_chart(
                grafico_historico_previsao(dados, semanas, previsoes_sel, nome_sel, horizonte),
                use_container_width=True
            )

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            st.plotly_chart(grafico_comparacao_metodos(dados, semanas, resultados_metodos), use_container_width=True)
        with col_g4:
            st.plotly_chart(grafico_tendencia(dados, semanas), use_container_width=True)

        # --------------------------------------------------------
        # COMPARAÇÃO E RANKING
        # --------------------------------------------------------
        st.markdown('<div class="section-header">🏆 Comparação dos Métodos</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="warning-box">⚠️ <b>Atenção:</b> O menor erro histórico não garante '
            'melhor previsão futura. Considere a interpretabilidade e a estabilidade do método.</div>',
            unsafe_allow_html=True
        )

        # Tabela de ranking
        linhas_rank = []
        for pos, (nome_r, _, _, erros_r) in enumerate(
            sorted(resultados_metodos, key=lambda x: (x[3][0] or 9999))
        ):
            mae_r, mape_r, rmse_r = erros_r
            icone = "🥇" if pos == 0 else ("🥈" if pos == 1 else ("🥉" if pos == 2 else ""))
            linhas_rank.append({
                "Pos.": f"{icone} {pos+1}º",
                "Método": nome_r,
                "MAE": f"{mae_r:.2f}" if mae_r else "—",
                "MAPE (%)": f"{mape_r:.2f}%" if mape_r else "—",
                "RMSE": f"{rmse_r:.2f}" if rmse_r else "—",
            })

        df_rank = pd.DataFrame(linhas_rank)
        st.dataframe(df_rank, use_container_width=True, hide_index=True)

        st.success(f"🏆 **Melhor Método Histórico:** {melhor_metodo_nome} (menor MAE)")

        # --------------------------------------------------------
        # ANÁLISE DA DEMANDA
        # --------------------------------------------------------
        st.markdown('<div class="section-header">🔍 Análise da Demanda</div>', unsafe_allow_html=True)

        tipo_tend, slope_val, cv_val = analisar_tendencia(dados)
        es_val, std_val = calcular_estoque_seguranca(dados)

        col_a, col_b, col_c, col_d = st.columns(4)
        label_tend = {
            "crescimento": ("📈 Crescimento", "#27AE60"),
            "queda": ("📉 Queda", "#E74C3C"),
            "estabilidade": ("⚖️ Estável", "#3498DB"),
            "alta_variabilidade": ("⚠️ Alta Variab.", "#F39C12")
        }
        tend_label, tend_cor = label_tend.get(tipo_tend, ("—", "#777"))

        with col_a:
            st.markdown(f"""<div class="metric-card" style="border-left-color:{tend_cor}">
                <div class="metric-label">Tendência</div>
                <div class="metric-value" style="font-size:1.3rem">{tend_label}</div>
                <div class="metric-sub">slope: {slope_val:+.2f} un/sem</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Coef. Variação</div>
                <div class="metric-value">{cv_val:.1f}%</div>
                <div class="metric-sub">{"Alta variabilidade" if cv_val > 25 else "Variabilidade normal"}</div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Desvio Padrão</div>
                <div class="metric-value">{std_val:.1f}</div>
                <div class="metric-sub">unidades/semana</div>
            </div>""", unsafe_allow_html=True)
        with col_d:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Estoque Segurança</div>
                <div class="metric-value">{es_val:.0f}</div>
                <div class="metric-sub">unidades (Z=1,65 / 95%)</div>
            </div>""", unsafe_allow_html=True)

        st.info(
            "💡 **Estoque de Segurança** garante que a empresa possa atender à demanda mesmo em semanas "
            "com vendas acima da média, evitando ruptura de estoque e perda de encomendas."
        )

        # --------------------------------------------------------
        # CAPACIDADE PRODUTIVA
        # --------------------------------------------------------
        st.markdown('<div class="section-header">🏭 Análise da Capacidade Produtiva</div>', unsafe_allow_html=True)

        if cap_prod > 0:
            dem_max_prev = max(previsoes_sel) if previsoes_sel else 0
            pct_uso = (dem_max_prev / cap_prod) * 100

            col_cp1, col_cp2, col_cp3 = st.columns([2, 2, 3])
            with col_cp1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Capacidade Semanal</div>
                    <div class="metric-value">{cap_prod}</div>
                    <div class="metric-sub">unidades/semana</div>
                </div>""", unsafe_allow_html=True)
            with col_cp2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Demanda Prevista Máx.</div>
                    <div class="metric-value">{dem_max_prev:.0f}</div>
                    <div class="metric-sub">unidades/semana</div>
                </div>""", unsafe_allow_html=True)
            with col_cp3:
                if pct_uso <= 85:
                    status_cap = "🟢 Capacidade Suficiente"
                    cor_cap = "#27AE60"
                elif pct_uso <= 100:
                    status_cap = "🟡 Próxima do Limite"
                    cor_cap = "#F39C12"
                else:
                    status_cap = "🔴 Capacidade Insuficiente"
                    cor_cap = "#E74C3C"

                st.markdown(f"""<div class="metric-card" style="border-left-color:{cor_cap}">
                    <div class="metric-label">Status</div>
                    <div class="metric-value" style="font-size:1.1rem;color:{cor_cap}">{status_cap}</div>
                    <div class="metric-sub">Utilização prevista: {pct_uso:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # Barra de utilização
            fig_cap = go.Figure(go.Bar(
                x=["Capacidade", "Demanda Prevista Máx."],
                y=[cap_prod, dem_max_prev],
                marker_color=["#2C3E50", cor_cap],
                text=[f"{cap_prod} un", f"{dem_max_prev:.0f} un"],
                textposition="outside"
            ))
            fig_cap.update_layout(
                title="Capacidade vs Demanda Prevista",
                yaxis_title="Unidades/Semana",
                template="plotly_white", height=320
            )
            st.plotly_chart(fig_cap, use_container_width=True)
        else:
            st.info("ℹ️ Informe a capacidade produtiva na barra lateral para ativar esta análise.")

        # --------------------------------------------------------
        # PREVISÕES FUTURAS (tabela)
        # --------------------------------------------------------
        st.markdown('<div class="section-header">🔮 Previsões Futuras</div>', unsafe_allow_html=True)

        df_prev = pd.DataFrame({
            "Semana Futura": [f"Semana {semanas[-1] + i + 1}" for i in range(horizonte)],
            "Demanda Prevista (un)": [f"{v:.0f}" for v in previsoes_sel],
            "Estoque Necessário (un)": [f"{v + es_val:.0f}" for v in previsoes_sel],
        })
        st.dataframe(df_prev, use_container_width=True, hide_index=True)

        # --------------------------------------------------------
        # INDICADORES DE ERRO DO MÉTODO SELECIONADO
        # --------------------------------------------------------
        st.markdown('<div class="section-header">📏 Indicadores de Erro – Método Selecionado</div>', unsafe_allow_html=True)

        mae_str  = f"{mae_sel:.2f}"  if mae_sel  else "N/A"
        mape_str = f"{mape_sel:.2f}%" if mape_sel else "N/A"
        rmse_str = f"{rmse_sel:.2f}" if rmse_sel else "N/A"

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">MAE – Erro Médio Absoluto</div>
                <div class="metric-value">{mae_str}</div>
                <div class="metric-sub">unidades de erro médio</div>
            </div>""", unsafe_allow_html=True)
        with col_e2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">MAPE – Erro % Médio</div>
                <div class="metric-value">{mape_str}</div>
                <div class="metric-sub">erro relativo médio</div>
            </div>""", unsafe_allow_html=True)
        with col_e3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">RMSE – Raiz Erro Quadrático</div>
                <div class="metric-value">{rmse_str}</div>
                <div class="metric-sub">penaliza erros grandes</div>
            </div>""", unsafe_allow_html=True)

        # --------------------------------------------------------
        # RECOMENDAÇÃO GERENCIAL
        # --------------------------------------------------------
        st.markdown('<div class="section-header">💼 Recomendação Gerencial</div>', unsafe_allow_html=True)

        rec_texto = gerar_recomendacao(
            tipo_tend, slope_val, cv_val, produto,
            cap_prod, previsoes_sel
        )

        st.markdown(f"""<div class="rec-box">
            <div class="rec-title">📋 Análise para: {produto}</div>
            <div class="rec-text">{rec_texto}</div>
        </div>""", unsafe_allow_html=True)

        # --------------------------------------------------------
        # DOWNLOAD DO RELATÓRIO PDF
        # --------------------------------------------------------
        st.markdown('<div class="section-header">📄 Relatório Gerencial</div>', unsafe_allow_html=True)

        if st.button("📄 Gerar e Baixar Relatório PDF", type="primary", use_container_width=False):
            with st.spinner("Gerando relatório PDF..."):
                pdf_bytes = gerar_pdf(
                    produto=produto,
                    dados=dados,
                    semanas=semanas,
                    metodo_nome=nome_sel,
                    previsoes=previsoes_sel,
                    mae=mae_sel,
                    mape=mape_sel,
                    rmse=rmse_sel,
                    tipo_tendencia=tipo_tend,
                    slope=slope_val,
                    cv=cv_val,
                    es=es_val,
                    std_demanda=std_val,
                    cap_prod=cap_prod,
                    recomendacao_texto=rec_texto
                )
                st.download_button(
                    label="⬇️ Baixar PDF",
                    data=pdf_bytes,
                    file_name="relatorio_demanda_madry.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ Relatório PDF gerado! Clique em 'Baixar PDF' acima.")

else:
    # Estado inicial sem dados
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #95A5A6;">
        <div style="font-size: 4rem;">📚</div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #2C3E50; margin-top: 1rem;">
            Bem-vindo ao Previsor de Demanda da Madry Papelaria Criativa
        </div>
        <div style="font-size: 0.95rem; margin-top: 0.5rem;">
            Insira o histórico de vendas acima para iniciar a análise.<br>
            <b>Exemplo:</b> 120,125,130,128,140,150,155,148,160,165,170,168
        </div>
    </div>
    """, unsafe_allow_html=True)
