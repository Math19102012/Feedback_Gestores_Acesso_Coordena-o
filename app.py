import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ===================== Configuração inicial =====================
st.set_page_config(page_title="Análise de Competências", layout="wide")
st.title("📈 Análise de Competências por Curso - FECAP")

# ===================== Carregamento de dados =====================
df_completo = pd.read_excel("data/df_limpo_avaliacoes.xlsx")

# ===================== Filtros (sidebar) =====================
st.sidebar.markdown("""
    <style>
    div[data-baseweb="checkbox"] input:checked ~ div {
        background-color: #228B22 !important;
        border-color: #228B22 !important;
    }
    </style>
    <h4 style='color:white;'>🎯 Filtros disponíveis</h4>
    """, unsafe_allow_html=True)

# Semestres
semestres_disponiveis = sorted(df_completo['Semestre'].dropna().unique(), reverse=True)
semestres_sel = []
with st.sidebar.expander("📅 Semestres", expanded=True):
    if st.checkbox("Selecionar todos os semestres", value=True, key="todos_semestres"):
        semestres_sel = semestres_disponiveis
    else:
        for semestre in semestres_disponiveis:
            if st.checkbox(semestre, value=False, key=semestre):
                semestres_sel.append(semestre)

# Cursos
cursos_disponiveis = sorted(df_completo['Curso (Matricula Atual) (Matricula)'].dropna().unique())
cursos_sel = []
with st.sidebar.expander("🏫 Cursos", expanded=True):
    if st.checkbox("Selecionar todos os cursos", value=True, key="todos_cursos"):
        cursos_sel = cursos_disponiveis
    else:
        for curso in cursos_disponiveis:
            if st.checkbox(curso, value=False, key=curso):
                cursos_sel.append(curso)

# ===================== Filtragem =====================
df_filtrado_geral = df_completo[
    (df_completo['Curso (Matricula Atual) (Matricula)'].notna()) &
    (df_completo['Semestre'].isin(semestres_sel))
].copy()

df_validos = df_filtrado_geral[df_filtrado_geral['Curso (Matricula Atual) (Matricula)'].isin(cursos_sel)].copy()

# Competências avaliadas
competencias = [
    "Conhecimentos", "Habilidades tecnicas", "Relacionamentos e parcerias",
    "Comunicação verbal e escrita", "Foco no cliente", "Gerenciamento do trabalho",
    "Orientação para resultados", "Aprendizagem pessoal"
]

# Long format
df_validos['Curso (Matricula Atual) (Matricula)'] = df_validos['Curso (Matricula Atual) (Matricula)'].astype(str)
df_long = pd.melt(
    df_validos,
    id_vars=['Curso (Matricula Atual) (Matricula)', 'Modelo'],
    value_vars=competencias,
    var_name='Competência',
    value_name='Nota'
)
df_long['Nota'] = df_long['Nota'].astype(str)

# ===================== Indicador total =====================
st.markdown(f"<h5>📊 Total de Relatórios Filtrados: {len(df_validos)}</h5>", unsafe_allow_html=True)

# ===================== Gráfico 1: distribuição por curso =====================
st.subheader("Distribuição de Alunos por Curso")
fig1, ax1 = plt.subplots(figsize=(10, 4))
barras = sns.countplot(
    data=df_validos,
    y='Curso (Matricula Atual) (Matricula)',
    order=df_validos['Curso (Matricula Atual) (Matricula)'].value_counts().index,
    ax=ax1
)
ax1.set_title("Distribuição de Alunos por Curso")
ax1.set_xlabel("Total de Alunos")
ax1.set_ylabel("Curso")
for bar in barras.patches:
    width = bar.get_width()
    if width > 0:
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, f"{int(width)}",
                 va='center', fontsize=10, fontweight='bold', color='black')
st.pyplot(fig1, use_container_width=True)

# ===================== Funções auxiliares =====================
ORDEM_NOTAS = ['F', 'D', 'ND', 'NO']  # ordem fixa

def gerar_pivot(df):
    tabela = df.groupby(['Competência', 'Nota']).size().reset_index(name='Total')
    pivot = tabela.pivot(index='Competência', columns='Nota', values='Total').fillna(0)
    pivot = pivot.reindex(columns=ORDEM_NOTAS, fill_value=0)  # garante todas as colunas
    return pivot

# Pivots de contagem
df_total_long = pd.melt(
    df_filtrado_geral,
    id_vars=['Curso (Matricula Atual) (Matricula)', 'Modelo'],
    value_vars=competencias,
    var_name='Competência',
    value_name='Nota'
)
df_total_long['Nota'] = df_total_long['Nota'].astype(str)
df_total_pivot = gerar_pivot(df_total_long)
df_curso_pivot = gerar_pivot(df_long)

# Converte para % verdadeiras (para rótulos e referência)
def to_percent(df_counts):
    pct = df_counts.div(df_counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    return pct.fillna(0)

df_total_pct_true = to_percent(df_total_pivot)
df_curso_pct_true = to_percent(df_curso_pivot)

# ========== ALOCAÇÃO VISUAL POR PIXELS (piso por faixa de % sem alterar labels) ==========
# Mapa de pisos visuais (px) por faixa de % (1..5). Ajuste como quiser:
MIN_PX_MAP = {1: 12, 2: 14, 3: 16, 4: 18, 5: 20}  # 1%→10px, 2%→12px, ... 5%→18px
THRESHOLD_PCT = max(MIN_PX_MAP.keys())            # 5
MIN_LABEL_PCT = 1.0                               # mostra label >= 1%

def min_px_for_pct(p):
    if p <= 0:
        return 0.0
    k = int(np.ceil(min(p, THRESHOLD_PCT)))
    # se p<=threshold usa o piso; acima de threshold, piso=0 (será proporcional)
    return MIN_PX_MAP.get(k, 0.0) if p <= THRESHOLD_PCT else 0.0

def allocate_pixels_for_df(df_pct_true, axis_px_height):
    """
    Para cada barra (linha), reserva pisos em pixels para porcentagens pequenas
    e redistribui os pixels restantes proporcionalmente às % verdadeiras dos
    segmentos > THRESHOLD_PCT. Retorna um DF de "alturas visuais" em % (somando 100).
    """
    cols = df_pct_true.columns
    vis_rows = []
    for _, row in df_pct_true.iterrows():
        p = row.values.astype(float)
        # pisos em px para quem tem p>0 e p<=threshold
        min_px = np.array([min_px_for_pct(pi) for pi in p], dtype=float)
        sum_min = min_px.sum()
        # se, por algum motivo, os pisos ultrapassarem a altura da barra, escala os pisos
        if sum_min > axis_px_height and sum_min > 0:
            min_px *= (axis_px_height / sum_min)
            sum_min = axis_px_height

        remaining_px = max(axis_px_height - sum_min, 0.0)

        # pesos só para os segmentos acima do limiar
        mask_above = p > THRESHOLD_PCT
        weights = p * mask_above
        sum_w = weights.sum()

        alloc_px = min_px.copy()
        if remaining_px > 0:
            if sum_w > 0:
                alloc_px += remaining_px * (weights / sum_w)
            else:
                # nenhum > limiar: distribui restante proporcional às próprias % >0 (ou por igual)
                positive = p > 0
                sum_pos = p[positive].sum()
                if sum_pos > 0:
                    alloc_px[positive] += remaining_px * (p[positive] / sum_pos)
                else:
                    # linha toda zero
                    pass

        # converte px alocados para "unidades de dados" (0..100) preservando soma=100
        vis_pct = (alloc_px / axis_px_height) * 100.0
        # Pode haver ruído numérico: normaliza para somar 100 exatamente
        s = vis_pct.sum()
        if s > 0:
            vis_pct = vis_pct * (100.0 / s)
        vis_rows.append(vis_pct)

    return pd.DataFrame(vis_rows, index=df_pct_true.index, columns=cols)

# ===================== Gráfico 2 (desenhado manualmente) =====================
# ===== Subtítulo + legenda com totais =====
def fmt_br(n: int) -> str:
    # milhar com ponto (pt-BR)
    return f"{n:,}".replace(",", ".")

total_verde = int(len(df_filtrado_geral))  # semestres selecionados (soma todos os cursos)
total_azul  = int(len(df_validos))         # semestres + cursos selecionados

st.markdown(f"""
<h3>Comparativo de Notas por Competência</h3>

<div style='font-size:16px;'>
  <span style='display:inline-block; width:15px; height:15px; background-color:#0D7C3B; margin-right:8px; border-radius:3px;'></span>
  <strong>Cores verdes:</strong> Total dos Semestres Selecionados (Considera Todos os Cursos) — 
  <strong>{fmt_br(total_verde)}</strong> relatórios
</div>

<div style='font-size:16px; margin-top:6px;'>
  <span style='display:inline-block; width:15px; height:15px; background-color:#04517e; margin-right:8px; border-radius:3px;'></span>
  <strong>Cores azuis:</strong> Total dos Semestres Selecionados (Considera Somente os Cursos Selecionados) — 
  <strong>{fmt_br(total_azul)}</strong> relatórios
</div>
""", unsafe_allow_html=True)


# Cores
cores_total = ['#66c2a5', '#41ae76', '#238b45', "#0D7C3B"]  # F, D, ND, NO (verde)
cores_curso = ["#8fafda", "#74acd4", "#278ec2", "#04517e"]  # F, D, ND, NO (azul)

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.set_ylim(0, 100)
ax2.set_xlabel("Competência")
ax2.set_ylabel("Proporção (%)")
ax2.set_title("Comparativo de Notas por Competência - Cursos Selecionados")

# Precisamos da altura do eixo em pixels para converter px→%
fig2.canvas.draw()  # necessário para obter medidas corretas
axis_px_height = ax2.get_window_extent(fig2.canvas.get_renderer()).height

# Calcula alturas VISUAIS (em %) a partir dos pisos em pixels
df_total_vis = allocate_pixels_for_df(df_total_pct_true, axis_px_height)
df_curso_vis = allocate_pixels_for_df(df_curso_pct_true, axis_px_height)

# Desenha barras empilhadas manualmente (para controlar posições/cores/labels)
indices = np.arange(len(df_curso_vis.index))
bar_width = 0.4
x_curso = indices - bar_width/2
x_total = indices + bar_width/2

def label_pct(p):
    # Se tiver valor positivo menor que 1%, mostramos "1%"
    if p > 0 and p < 1:
        return "1%"
    # Caso contrário, arredonda normalmente para 0 casas
    return f"{np.round(p):.0f}%"

# --- Desenho Cursos (azul) ---
bottom = np.zeros(len(indices))
for j, col in enumerate(ORDEM_NOTAS):
    h = df_curso_vis[col].values
    ax2.bar(x_curso, h, bar_width, bottom=bottom, color=cores_curso[j], edgecolor='none')

    # rótulos com % VERDADEIRAS, mostrando "1%" quando 0<p<1
    p_true = df_curso_pct_true[col].values
    y = bottom + h/2
    for i in range(len(indices)):
        if h[i] > 0 and p_true[i] > 0:
            y_i = min(y[i], 99.3)  # anti-clipping perto do topo
            ax2.text(
                x_curso[i], y_i, label_pct(p_true[i]),
                ha="center", va="center", fontsize=8, color="black", clip_on=False
            )
    bottom += h

# --- Desenho Total (verde) ---
bottom = np.zeros(len(indices))
for j, col in enumerate(ORDEM_NOTAS):
    h = df_total_vis[col].values
    ax2.bar(x_total, h, bar_width, bottom=bottom, color=cores_total[j], edgecolor='none')

    p_true = df_total_pct_true[col].values
    y = bottom + h/2
    for i in range(len(indices)):
        if h[i] > 0 and p_true[i] > 0:
            y_i = min(y[i], 99.3)  # anti-clipping perto do topo
            ax2.text(
                x_total[i], y_i, label_pct(p_true[i]),
                ha="center", va="center", fontsize=8, color="black", clip_on=False
            )
    bottom += h
# Eixos e legendas
ax2.set_xticks(indices)
ax2.set_xticklabels(df_curso_vis.index, rotation=45, ha='right')

# Legenda personalizada (8 itens: Cursos e Total)
legend_patches = [
    Patch(facecolor=cores_curso[0], label='F - Cursos'),
    Patch(facecolor=cores_curso[1], label='D - Cursos'),
    Patch(facecolor=cores_curso[2], label='ND - Cursos'),
    Patch(facecolor=cores_curso[3], label='NO - Cursos'),
    Patch(facecolor=cores_total[0], label='F - Total'),
    Patch(facecolor=cores_total[1], label='D - Total'),
    Patch(facecolor=cores_total[2], label='ND - Total'),
    Patch(facecolor=cores_total[3], label='NO - Total'),
]
ax2.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', title='Nota', borderaxespad=0)

# Margem para não cortar legenda
fig2.tight_layout(rect=(0, 0, 0.86, 1))
fig2.subplots_adjust(right=0.86)

st.pyplot(fig2, use_container_width=True)

# ===================== Download da base filtrada =====================
st.subheader("⬇️ Baixar dados utilizados")
def converter_para_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Dados")
    output.seek(0)
    return output

st.download_button(
    label="📥 Baixar base consolidada (.xlsx)",
    data=converter_para_excel(df_validos.copy()),
    file_name="avaliacoes_competencias_filtrado.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ===================== Detalhe por Competência e Nota =====================
st.subheader("🔍 Detalhar alunos por Competência e Nota")
col1, col2 = st.columns(2)
with col1:
    competencia_sel = st.selectbox("📘 Selecione a competência:", sorted(df_long['Competência'].unique()))
with col2:
    nota_sel = st.selectbox("🏷️ Selecione a nota:", sorted(df_long['Nota'].unique()))

detalhe = df_long[
    (df_long['Competência'] == competencia_sel) &
    (df_long['Nota'] == nota_sel)
]

if not detalhe.empty:
    st.markdown(f"*Alunos com nota '{nota_sel}' em '{competencia_sel}':*")
    st.dataframe(detalhe[['Curso (Matricula Atual) (Matricula)', 'Competência', 'Nota']])
else:
    st.info("Nenhum aluno encontrado com essa combinação.")
