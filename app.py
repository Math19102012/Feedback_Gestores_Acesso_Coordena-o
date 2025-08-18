import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Análise de Competências", layout="wide")
st.title("📈 Análise de Competências por Curso - FECAP")

# Carrega os dados
df_completo = pd.read_excel("data/df_limpo_avaliacoes.xlsx")

# --- FILTROS ---
st.sidebar.markdown("""
    <style>
    div[data-baseweb="checkbox"] input:checked ~ div {
        background-color: #228B22 !important;
        border-color: #228B22 !important;
    }
    </style>
    <h4 style='color:white;'>🎯 Filtros disponíveis</h4>
    """, unsafe_allow_html=True)

# Semestres com checkboxes
semestres_disponiveis = sorted(df_completo['Semestre'].dropna().unique(), reverse=True)
semestres_sel = []
with st.sidebar.expander("📅 Semestres", expanded=True):
    if st.checkbox("Selecionar todos os semestres", value=True, key="todos_semestres"):
        semestres_sel = semestres_disponiveis
    else:
        for semestre in semestres_disponiveis:
            if st.checkbox(semestre, value=False, key=semestre):
                semestres_sel.append(semestre)

# Cursos com checkboxes
cursos_disponiveis = sorted(df_completo['Curso (Matricula Atual) (Matricula)'].dropna().unique())
cursos_sel = []
with st.sidebar.expander("🏫 Cursos", expanded=True):
    if st.checkbox("Selecionar todos os cursos", value=True, key="todos_cursos"):
        cursos_sel = cursos_disponiveis
    else:
        for curso in cursos_disponiveis:
            if st.checkbox(curso, value=False, key=curso):
                cursos_sel.append(curso)

# --- FILTRAGEM ---
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

# Reorganiza os dados para formato longo
df_validos['Curso (Matricula Atual) (Matricula)'] = df_validos['Curso (Matricula Atual) (Matricula)'].astype(str)
df_long = pd.melt(df_validos, id_vars=['Curso (Matricula Atual) (Matricula)', 'Modelo'], 
                  value_vars=competencias, var_name='Competência', value_name='Nota')
df_long['Nota'] = df_long['Nota'].astype(str)

# Mostra quantidade total de relatórios filtrados
st.markdown(f"<h5 '>📊 Total de Relatórios Filtrados: {len(df_validos)}</h5>", unsafe_allow_html=True)

# Gráfico de distribuição por curso
st.subheader("Distribuição de Alunos por Curso")
fig1, ax1 = plt.subplots(figsize=(10, 4))
barras = sns.countplot(data=df_validos, y='Curso (Matricula Atual) (Matricula)',
                       order=df_validos['Curso (Matricula Atual) (Matricula)'].value_counts().index, ax=ax1)
ax1.set_title("Distribuição de Alunos por Curso")
ax1.set_xlabel("Total de Alunos")
ax1.set_ylabel("Curso")
for bar in barras.patches:
    height = bar.get_width()
    if height > 0:
        ax1.text(height + 1, bar.get_y() + bar.get_height() / 2, f'{int(height)}',
                 va='center', fontsize=10, fontweight='bold', color='black')
st.pyplot(fig1)


# Função para gerar pivot
def gerar_pivot(df):
    tabela = df.groupby(['Competência', 'Nota']).size().reset_index(name='Total')
    pivot = tabela.pivot(index='Competência', columns='Nota', values='Total').fillna(0)
    ordem_notas = ['F', 'D', 'ND', 'NO']
    pivot = pivot[[n for n in ordem_notas if n in pivot.columns]]
    return pivot


# Pivot total (com todos os cursos dos semestres selecionados)
df_total_long = pd.melt(df_filtrado_geral, id_vars=['Curso (Matricula Atual) (Matricula)', 'Modelo'], 
                        value_vars=competencias, var_name='Competência', value_name='Nota')
df_total_long['Nota'] = df_total_long['Nota'].astype(str)
df_total_pivot = gerar_pivot(df_total_long)

# Pivot dos cursos filtrados
df_curso_pivot = gerar_pivot(df_long)

# Gráfico comparativo
fig2, ax2 = plt.subplots(figsize=(12, 6))
bar_width = 0.4

# Tons de verde para "Total"
verde_palette = ListedColormap(['#66c2a5', '#41ae76', '#238b45', "#0D7C3B"])

# Tons de azul para "Cursos Selecionados"
azul_palette = ListedColormap(["#8fafda", "#74acd4", "#278ec2", "#04517e"])

# Sub Titulo - Grafico Dois
st.markdown("""
<h3 >
    Comparativo de Notas por Competência
</h3>

<div style='font-size:16px;'>
    <span style='display:inline-block; width:15px; height:15px; background-color:#0D7C3B; margin-right:8px; border-radius:3px;'></span>
    <strong>Cores verdes:</strong> Total dos Semestres selecionados (Soma Todos os Cursos)
</div>

<div style='font-size:16px; margin-top:6px;'>
    <span style='display:inline-block; width:15px; height:15px; background-color:#04517e; margin-right:8px; border-radius:3px;'></span>
    <strong>Cores azuis:</strong> Total dos cursos selecionados
</div>
""", unsafe_allow_html=True)

# Curso (esquerda) e Total (direita)
df_curso_pivot.plot(kind='bar', stacked=True, ax=ax2, position=0, colormap=azul_palette, width=bar_width, label='Cursos')
df_total_pivot.plot(kind='bar', stacked=True, ax=ax2, position=1, colormap=verde_palette, width=bar_width, label='Total')
ax2.set_title("Comparativo de Notas por Competência - Cursos Selecionados")
ax2.set_xlabel("Competência")
ax2.set_ylabel("Total de Avaliações")
ax2.legend(title='Nota')

# Função de anotação

def adicionar_texto(ax, i, offset, cumulative, valor, pequeno_count):
    if valor >= 5:
        y_pos = cumulative + valor / 2
        ax.text(i + offset, y_pos, int(valor), ha='center', va='center', fontsize=9, color='black')
    else:
        y_bar_top = cumulative + valor
        if pequeno_count == 0:
            deslocamento_externo = 0.30
            x_text = i + offset + deslocamento_externo
            ax.text(x_text, y_bar_top, int(valor), ha='center', va='center', fontsize=9, color='black')
            ax.annotate("", xy=(i + offset + (bar_width/2 if offset>0 else -bar_width/2), y_bar_top),
                        xytext=(x_text, y_bar_top),
                        arrowprops=dict(arrowstyle="-", linestyle='dashed', color='black', lw=0.8))
        else:
            y_pos = cumulative + valor / 2
            ax.text(i + offset, y_pos, int(valor), ha='center', va='center', fontsize=9, color='black')

# Adiciona números nas barras
for i, competencia in enumerate(df_curso_pivot.index):
    cumulative = 0
    pequenos = 0
    for nota in df_curso_pivot.columns:
        valor = df_curso_pivot.loc[competencia, nota]
        if valor > 0:
            adicionar_texto(ax2, i, bar_width/2, cumulative, valor, pequenos)
            if valor < 3:
                pequenos += 1
            cumulative += valor

for i, competencia in enumerate(df_total_pivot.index):
    cumulative = 0
    pequenos = 0
    for nota in df_total_pivot.columns:
        valor = df_total_pivot.loc[competencia, nota]
        if valor > 0:
            adicionar_texto(ax2, i, -bar_width/2, cumulative, valor, pequenos)
            if valor < 3:
                pequenos += 1
            cumulative += valor

st.pyplot(fig2)

# Exportar os dados
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

# Bloco detalhado de alunos por competência
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