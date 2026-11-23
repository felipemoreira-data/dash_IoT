from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, no_update
from plotly.subplots import make_subplots


CSV = Path(__file__).resolve().parent / "data" / "leituras_tomada_iot_finais.csv"
BRASILIA = timezone(timedelta(hours=-3))

AZUL = "#245dd8"
VERDE = "#087f74"
ROXO = "#7660b4"
CONFIG_GRAFICO = {"displaylogo": False, "responsive": True}


def carregar_dados():
    dados = pd.read_csv(CSV, encoding="utf-8-sig")
    campos_numericos = ["potencia_w", "tensao_v", "corrente_ma"]
    obrigatorias = ["timestamp_coleta", "device_id", *campos_numericos]

    if dados.empty or not set(obrigatorias).issubset(dados.columns):
        raise ValueError(
            "O CSV deve conter leituras com horário, dispositivo, "
            "potência, tensão e corrente."
        )

    horario = pd.to_datetime(dados["timestamp_coleta"], format="mixed", utc=True)
    dados["horario"] = horario.dt.tz_convert(BRASILIA).dt.tz_localize(None)

    for campo in campos_numericos:
        dados[campo] = pd.to_numeric(dados[campo], errors="raise")

    if dados[[*obrigatorias, "horario"]].isna().any().any():
        raise ValueError("Há campos obrigatórios vazios no CSV.")
    if dados.duplicated(["horario", "device_id"]).any():
        raise ValueError("Há horários duplicados para o mesmo dispositivo.")
    if dados["device_id"].nunique() != 1:
        raise ValueError("O painel espera as leituras de uma única tomada.")

    dados = dados.sort_values("horario")
    dados["data"] = dados["horario"].dt.strftime("%Y-%m-%d")
    return dados


DADOS = carregar_dados()
DATAS = sorted(DADOS["data"].unique())


def data_br(data):
    return pd.Timestamp(data).strftime("%d/%m/%Y")


def numero(valor, casas=0):
    return f"{valor:,.{casas}f}".translate(str.maketrans(".,", ",."))


def cartao(rotulo, valor):
    return html.Div(
        [html.Span(rotulo), html.Strong(valor)],
        className="indicador",
    )


def estilo(figura, altura):
    figura.update_layout(
        template="plotly_white",
        height=altura,
        margin=dict(l=60, r=24, t=24, b=78),
        font=dict(family="system-ui, sans-serif", color="#263852", size=13),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        separators=",.",
        hoverlabel=dict(font_size=14),
    )
    figura.update_yaxes(gridcolor="#e6edf5", zeroline=False, automargin=True)
    figura.update_xaxes(automargin=True)
    return figura


def graficos_resumo(dados):
    barras = estilo(go.Figure(), 355)
    caixas = estilo(go.Figure(), 360)

    if dados.empty:
        for figura in (barras, caixas):
            figura.add_annotation(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text="Nenhuma leitura no período selecionado.",
                showarrow=False,
            )
            figura.update_xaxes(visible=False)
            figura.update_yaxes(visible=False)
        return barras, caixas

    diario = dados.groupby("data", as_index=False).agg(
        media=("potencia_w", "mean"),
        amostras=("potencia_w", "size"),
        inicio=("horario", "min"),
        fim=("horario", "max"),
    )
    rotulos = [data_br(data) for data in diario["data"]]
    informacoes = list(zip(
        diario["data"],
        diario["amostras"],
        diario["inicio"].dt.strftime("%H:%M:%S"),
        diario["fim"].dt.strftime("%H:%M:%S"),
    ))

    barras.add_bar(
        x=rotulos,
        y=diario["media"],
        customdata=informacoes,
        marker_color=AZUL,
        hovertemplate=(
            "%{x}<br>Média: %{y:.1f} W"
            "<br>Leituras: %{customdata[1]}"
            "<br>Janela: %{customdata[2]} – %{customdata[3]}<extra></extra>"
        ),
    )
    barras.update_layout(clickmode="event", hovermode="closest")
    barras.update_yaxes(rangemode="tozero")

    caixas.add_box(
        x=[data_br(data) for data in dados["data"]],
        y=dados["potencia_w"],
        name="Potência",
        marker_color=VERDE,
        boxpoints="outliers",
        showlegend=False,
    )

    for figura in (barras, caixas):
        figura.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=rotulos,
            tickangle=-45,
        )
        figura.update_yaxes(title_text="Potência (W)")

    return barras, caixas


def graficos_detalhe(data):
    dados = DADOS[DADOS["data"] == data]
    if dados.empty:
        return estilo(go.Figure(), 540), "Selecione uma data com leituras."

    figura = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.44, 0.28, 0.28],
    )
    series = [
        ("potencia_w", "Potência", "W", AZUL, ".1f"),
        ("tensao_v", "Tensão", "V", VERDE, ".2f"),
        ("corrente_ma", "Corrente", "mA", ROXO, ".0f"),
    ]

    # Interrompe as linhas quando há uma lacuna na sequência de leituras.
    intervalos = dados["horario"].diff()
    intervalo_tipico = intervalos[intervalos > pd.Timedelta(0)].median()
    segmentos = dados.groupby((intervalos > 3 * intervalo_tipico).cumsum(), sort=False)

    for linha, (campo, rotulo, unidade, cor, casas) in enumerate(series, start=1):
        for _, trecho in segmentos:
            figura.add_trace(
                go.Scatter(
                    x=trecho["horario"],
                    y=trecho[campo],
                    name=rotulo,
                    mode="lines+markers",
                    line=dict(color=cor, width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate=(
                        f"%{{x|%H:%M:%S}}<br>{rotulo}: "
                        f"%{{y:{casas}}} {unidade}<extra></extra>"
                    ),
                ),
                row=linha,
                col=1,
            )
        figura.update_yaxes(title_text=f"{rotulo} ({unidade})", row=linha, col=1)

    media = dados["potencia_w"].mean()
    figura.add_hline(
        y=media,
        row=1,
        col=1,
        line_dash="dot",
        line_color="#64748b",
        annotation_text=f"Média: {numero(media, 1)} W",
    )
    figura.update_xaxes(tickformat="%H:%M", hoverformat="%H:%M:%S", type="date")
    figura.update_xaxes(title_text="Horário de Brasília", row=3, col=1)
    figura.update_layout(hovermode="x")

    inicio, fim = dados["horario"].iloc[[0, -1]]
    legenda = (
        f"{numero(len(dados))} leituras · "
        f"{inicio:%H:%M:%S} a {fim:%H:%M:%S} · horário de Brasília"
    )
    return estilo(figura, 665), legenda


app = Dash(__name__, title="Energia em dados", update_title=None)
server = app.server

app.layout = html.Main([
    html.Header([
        html.P("MONITORAMENTO COM TOMADA INTELIGENTE"),
        html.H1("Energia em dados"),
        html.Div("Acompanhe a potência, a tensão e a corrente do ventilador."),
    ], className="cabecalho"),

    dcc.Tabs(id="abas", value="visao", className="abas", children=[
        dcc.Tab(label="Visão geral", value="visao"),
        dcc.Tab(label="Detalhe por data", value="detalhe"),
    ]),

    html.Section(id="visao", children=[
        html.Div([
            html.Div([
                html.H2("Visão geral"),
                html.P("Compare as janelas de registro por dia."),
            ]),
            html.Div([
                html.Label("Período", htmlFor="periodo"),
                dcc.DatePickerRange(
                    id="periodo",
                    min_date_allowed=DATAS[0],
                    max_date_allowed=DATAS[-1],
                    start_date=DATAS[0],
                    end_date=DATAS[-1],
                    display_format="DD/MM/YYYY",
                    first_day_of_week=0,
                ),
            ], className="controle"),
        ], className="barra"),

        html.Div(id="indicadores", className="indicadores"),

        html.Section([
            html.H3("Potência média registrada por dia"),
            html.P("Clique em uma barra para abrir as leituras daquela data."),
            dcc.Graph(id="barras", config=CONFIG_GRAFICO),
        ], className="grafico"),

        html.Section([
            html.H3("Distribuição da potência por dia"),
            html.P(
                "A linha dentro da caixa marca a mediana. "
                "A caixa cobre os 50% centrais das leituras."
            ),
            dcc.Graph(id="caixas", config=CONFIG_GRAFICO),
        ], className="grafico"),
    ]),

    html.Section(id="detalhe", hidden=True, children=[
        html.Div([
            html.Div([
                html.H2("Detalhe por data"),
                html.P(id="janela"),
            ]),
            html.Div([
                html.Label("Data das leituras", htmlFor="data"),
                dcc.Dropdown(
                    id="data",
                    value=DATAS[0],
                    clearable=False,
                    options=[{"label": data_br(data), "value": data} for data in DATAS],
                ),
            ], className="controle data"),
        ], className="barra"),

        html.Section([
            html.H3("Potência, tensão e corrente ao longo da sessão"),
            dcc.Graph(id="series", config=CONFIG_GRAFICO),
        ], className="grafico"),
    ]),

    html.Footer(
        "Indicadores calculados sobre as leituras disponíveis nas janelas de registro."
    ),
])


@callback(
    Output("indicadores", "children"),
    Output("barras", "figure"),
    Output("caixas", "figure"),
    Input("periodo", "start_date"),
    Input("periodo", "end_date"),
)
def atualizar_visao(inicio, fim):
    dados = DADOS[DADOS["data"].between(inicio or DATAS[0], fim or DATAS[-1])]
    pico = f"{numero(dados['potencia_w'].max(), 1)} W" if not dados.empty else "—"
    indicadores = [
        cartao("Leituras", numero(len(dados))),
        cartao("Datas com registros", numero(dados["data"].nunique())),
        cartao("Maior potência observada", pico),
    ]
    barras, caixas = graficos_resumo(dados)
    return indicadores, barras, caixas


@callback(
    Output("abas", "value"),
    Output("data", "value"),
    Input("barras", "clickData"),
    prevent_initial_call=True,
)
def abrir_data(clique):
    if not clique or not clique.get("points"):
        return no_update, no_update
    return "detalhe", clique["points"][0]["customdata"][0]


@callback(
    Output("visao", "hidden"),
    Output("detalhe", "hidden"),
    Input("abas", "value"),
)
def mostrar_aba(aba):
    return aba != "visao", aba != "detalhe"


@callback(
    Output("series", "figure"),
    Output("janela", "children"),
    Input("data", "value"),
    Input("abas", "value"),
)
def atualizar_detalhe(data, aba):
    if aba == "detalhe":
        return graficos_detalhe(data)
    return no_update, no_update


if __name__ == "__main__":
    app.run(debug=False)