"""
Analisador de Reclamações - Flask + NLTK + Pandas (arquivo único)

Objetivo: ajudar a empresa a entender as reclamações de clientes.

Recursos:
1. UPLOAD de planilha (.csv ou .xlsx) no topo da tela. Os dados são
   processados automaticamente e viram 3 gráficos de pizza interativos:
   Categoria/Motivo, Status e Canal de atendimento.
2. ANÁLISE DE TEXTO: o funcionário cola uma reclamação e vê as palavras
   mais frequentes (NLTK).

Rodar localmente:   python app.py   (abre o navegador sozinho)
Rodar no Render:    gunicorn app:app
"""

import io
import os
import sys
import threading
import unicodedata
import webbrowser
from collections import Counter

import nltk
import pandas as pd
from flask import Flask, redirect, render_template_string, request
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Proteção contra o comando errado: este arquivo é FLASK, não Streamlit.
if "streamlit" in sys.modules:
    raise RuntimeError(
        "Este arquivo é um app Flask. Pare o Streamlit (Ctrl+C) e rode: python app.py"
    )

# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÕES
# ---------------------------------------------------------------------------
LIMITE_CARACTERES = 20000          # tamanho máximo do texto digitado
OPCOES_TOP = (5, 10, 15, 20)       # quantidades que o usuário pode escolher
TOP_PADRAO = 10                    # valor inicial da lista
TAMANHO_MINIMO = 3                 # ignora palavras com menos de 3 letras

LIMITE_ARQUIVO_MB = 10             # tamanho máximo da planilha enviada
EXTENSOES = (".csv", ".xlsx")      # formatos aceitos no upload
MAX_FATIAS = 8                     # fatias por gráfico; o resto vira "Outros"

# Os 3 gráficos. "palavras" são trechos procurados no NOME das colunas da
# planilha (sem acento e em minúsculas), do mais forte ao mais fraco.
GRAFICOS = {
    "categoria": {
        "titulo": "Categoria / Motivo",
        "palavras": ("categ", "motivo", "assunto", "tipo"),
    },
    "status": {
        "titulo": "Status (Resolvida / Pendente)",
        "palavras": ("status", "situac", "resolvida", "resolvido"),
    },
    "canal": {
        "titulo": "Canal de atendimento",
        "palavras": ("canal", "origem", "meio de contato", "meio de atendimento"),
    },
}
# Ordem de busca das colunas: as de nome mais específico vêm primeiro
ORDEM_DETECCAO = ("status", "canal", "categoria")

# Cores das fatias (a ordem de cores se repete se houver mais fatias)
PALETA = ["#2563EB", "#F59E0B", "#10B981", "#EF4444",
          "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16"]
CORES_FIXAS = {
    "Resolvida": "#10B981",
    "Pendente": "#F59E0B",
    "Outros": "#94A3B8",
    "Não informado": "#CBD5E1",
}

# Palavras comuns em reclamações que não ajudam a achar o problema.
# Ajuste esta lista conforme o negócio da empresa.
STOPWORDS_EXTRAS = {"pra", "pro", "vcs", "aqui", "ali", "assim", "então"}

# O NLTK guardará os dados baixados dentro da pasta do projeto
PASTA_NLTK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
nltk.data.path.insert(0, PASTA_NLTK)

# Texto de exemplo para o botão "Usar texto de exemplo"
EXEMPLO = (
    "O produto chegou com atraso e a embalagem estava danificada. "
    "Entrei em contato com o atendimento três vezes e ninguém resolveu o "
    "problema. A qualidade do produto é péssima, quebrou em uma semana. "
    "O atendimento demorou para responder e o prazo de troca é muito longo. "
    "Estou decepcionado com a qualidade e com o atendimento."
)

app = Flask(__name__)
# O Flask recusa automaticamente arquivos maiores que este limite
app.config["MAX_CONTENT_LENGTH"] = LIMITE_ARQUIVO_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# 2. PROCESSAMENTO DE TEXTO (NLTK)
# ---------------------------------------------------------------------------
def remover_acentos(palavra: str) -> str:
    """Troca 'não' por 'nao', 'você' por 'voce' etc."""
    decomposta = unicodedata.normalize("NFD", palavra)
    return "".join(c for c in decomposta if unicodedata.category(c) != "Mn")


def carregar_stopwords() -> set:
    """Carrega as stopwords em português; baixa do NLTK se ainda não existirem."""
    try:
        base = set(stopwords.words("portuguese"))
    except LookupError:
        # Primeira execução (ex.: servidor novo do Render): baixa a lista
        nltk.download("stopwords", download_dir=PASTA_NLTK, quiet=True)
        base = set(stopwords.words("portuguese"))

    base |= STOPWORDS_EXTRAS
    # Também guardamos a versão sem acento, pois muita gente digita "nao"
    # em vez de "não" nas reclamações.
    return base | {remover_acentos(p) for p in base}


STOPWORDS_PT = carregar_stopwords()

# Tokenizador: pega apenas sequências de letras (com acentos), ignorando
# números e pontuação. Não depende do pacote "punkt".
TOKENIZADOR = RegexpTokenizer(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")


def analisar_texto(texto: str, top_n: int) -> dict:
    """Conta as palavras mais frequentes do texto.

    Passos: minúsculas -> separar palavras -> remover stopwords e palavras
    curtas -> contar com FreqDist -> pegar as N mais comuns.
    """
    palavras = TOKENIZADOR.tokenize(texto.lower())

    # Mantém só palavras com significado para a análise
    relevantes = [
        p for p in palavras
        if len(p) >= TAMANHO_MINIMO and p not in STOPWORDS_PT
    ]

    frequencia = FreqDist(relevantes)          # dicionário palavra -> quantidade
    mais_comuns = frequencia.most_common(top_n)

    maior = mais_comuns[0][1] if mais_comuns else 1
    total = len(relevantes)
    ranking = [
        {
            "posicao": i,
            "palavra": palavra,
            "quantidade": qtd,
            "largura": round(qtd / maior * 100),         # tamanho da barra
            "participacao": f"{qtd / total:.1%}",        # % entre as relevantes
        }
        for i, (palavra, qtd) in enumerate(mais_comuns, start=1)
    ]
    return {
        "ranking": ranking,
        "total_palavras": len(palavras),
        "total_analisadas": total,
        "total_diferentes": len(frequencia),
    }


# ---------------------------------------------------------------------------
# 3. PROCESSAMENTO DE PLANILHAS (PANDAS)
# ---------------------------------------------------------------------------
def normalizar_nome(texto) -> str:
    """Deixa o nome da coluna em minúsculas e sem acento: 'Situação' -> 'situacao'."""
    return remover_acentos(str(texto)).strip().lower()


def ler_planilha(arquivo) -> pd.DataFrame:
    """Lê um .csv ou .xlsx enviado pelo usuário, direto da memória."""
    conteudo = arquivo.read()

    if arquivo.filename.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(conteudo), engine="openpyxl")

    # CSV: tenta UTF-8 e depois Latin-1 (comum em planilhas brasileiras).
    # sep=None deixa o pandas descobrir se o separador é vírgula ou ";".
    for codificacao in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(conteudo), sep=None,
                               engine="python", encoding=codificacao)
        except UnicodeDecodeError:
            continue
    raise ValueError("Codificação do CSV inválida.")


def detectar_colunas(df: pd.DataFrame) -> dict:
    """Descobre quais colunas da planilha são categoria, status e canal."""
    encontradas, usadas = {}, set()
    for chave in ORDEM_DETECCAO:
        # Para cada palavra-chave (da mais forte à mais fraca), procura uma
        # coluna cujo nome a contenha e que ainda não tenha sido usada.
        for palavra in GRAFICOS[chave]["palavras"]:
            if chave in encontradas:
                break
            for coluna in df.columns:
                if coluna not in usadas and palavra in normalizar_nome(coluna):
                    encontradas[chave] = coluna
                    usadas.add(coluna)
                    break
    return encontradas


def montar_distribuicao(serie: pd.Series, chave: str) -> dict:
    """Conta quantas reclamações há em cada valor e prepara os dados do gráfico."""
    # Agrupa valores iguais ignorando maiúsculas, acentos e espaços
    # ("Atraso", "atraso " e "ATRASO" viram um grupo só).
    grupos = {}
    for valor in serie:
        if pd.isna(valor) or str(valor).strip() == "":
            rotulo = "Não informado"
        else:
            rotulo = str(valor).strip()
        norma = remover_acentos(rotulo).lower()

        # No status, unifica "resolvido/resolvida" e "pendente/pendentes"
        if chave == "status":
            if norma.startswith("resolv"):
                norma, rotulo = "resolvida", "Resolvida"
            elif norma.startswith("pendent"):
                norma, rotulo = "pendente", "Pendente"

        grupos.setdefault(norma, Counter())[rotulo] += 1

    # Nome exibido = grafia mais comum do grupo. Ordena do maior para o menor.
    itens = [(c.most_common(1)[0][0], sum(c.values())) for c in grupos.values()]
    itens.sort(key=lambda item: item[1], reverse=True)

    # Muitas fatias deixam a pizza ilegível: o excedente vira "Outros"
    fatias = itens[:MAX_FATIAS]
    resto = itens[MAX_FATIAS:]
    if resto:
        fatias.append(("Outros", sum(q for _, q in resto)))

    total = sum(q for _, q in fatias)
    nomes = [nome for nome, _ in fatias]
    valores = [qtd for _, qtd in fatias]
    # Porcentagem com vírgula, no padrão brasileiro (ex.: 35,2)
    percentuais = [f"{qtd / total * 100:.1f}".replace(".", ",") for qtd in valores]

    return {
        "id": chave,
        "titulo": GRAFICOS[chave]["titulo"],
        "total": total,
        "nomes": nomes,
        "valores": valores,
        "percentuais": percentuais,
        # Texto da legenda: já inclui a porcentagem
        "legendas": [f"{n} ({p}%)" for n, p in zip(nomes, percentuais)],
        "cores": [CORES_FIXAS.get(n, PALETA[i % len(PALETA)])
                  for i, n in enumerate(nomes)],
    }


# ---------------------------------------------------------------------------
# 4. PÁGINA (HTML + CSS + JavaScript dentro do próprio arquivo)
# ---------------------------------------------------------------------------
PAGINA = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Análise de Reclamações</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #F1F3F6; color: #1F2937;
           font-family: "Segoe UI", Arial, sans-serif; }
    header { background: #1D4ED8; color: #fff; padding: 22px 16px; }
    header h1 { margin: 0; font-size: 22px; }
    header p { margin: 4px 0 0; font-size: 14px; opacity: .85; }
    main { max-width: 1100px; margin: 24px auto; padding: 0 16px; }
    .cartao { background: #fff; border-radius: 10px; padding: 24px;
              margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
    h2 { margin: 0 0 4px; font-size: 18px; }
    .dica { font-size: 13px; color: #64748B; margin: 0; }
    label { font-weight: 600; display: block; margin-bottom: 8px; }
    textarea { width: 100%; min-height: 200px; padding: 12px; resize: vertical;
               border: 1px solid #CBD5E1; border-radius: 8px; font: inherit; }
    textarea:focus, select:focus { outline: 2px solid #1D4ED8; }
    .linha { display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
             justify-content: space-between; margin-top: 14px; }
    select { padding: 8px 10px; border: 1px solid #CBD5E1; border-radius: 8px;
             font: inherit; }
    .contador { font-size: 13px; color: #64748B; }
    button, .botao-upload { border: 0; border-radius: 8px; padding: 10px 22px;
             font: inherit; font-weight: 600; cursor: pointer; }
    .principal, .botao-upload { background: #1D4ED8; color: #fff; }
    .principal:hover, .botao-upload:hover { background: #1E40AF; }
    .secundario { background: #E2E8F0; color: #1F2937; }

    /* Upload no topo */
    .upload { display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
              justify-content: space-between; border: 2px dashed #93C5FD;
              background: #EFF6FF; }
    .botao-upload { display: inline-block; font-size: 16px; padding: 14px 28px; }
    #arquivo { display: none; }
    #estado-upload { font-size: 14px; color: #1E40AF; font-weight: 600; }

    .alerta { background: #FEF2F2; color: #991B1B; border-left: 4px solid #DC2626;
              padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; }
    .aviso { background: #FFFBEB; color: #92400E; border-left: 4px solid #F59E0B;
             padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; }

    /* Gráficos de pizza */
    .resumo { font-size: 14px; color: #475569; margin-bottom: 12px; }
    .grade { display: grid; gap: 20px;
             grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .grade .cartao { margin-bottom: 0; }
    .area-grafico { position: relative; height: 360px; margin-top: 12px; }
    .espaco { height: 20px; }

    .metricas { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
    .metrica { flex: 1; min-width: 160px; background: #EFF6FF; border-radius: 8px;
               padding: 14px; text-align: center; }
    .metrica b { display: block; font-size: 24px; color: #1D4ED8; }
    .metrica span { font-size: 13px; color: #475569; }
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; font-size: 13px; color: #64748B; padding: 8px;
         border-bottom: 2px solid #E2E8F0; }
    td { padding: 10px 8px; border-bottom: 1px solid #E2E8F0; }
    .barra-fundo { background: #E2E8F0; border-radius: 6px; height: 14px; }
    .barra { background: #1D4ED8; border-radius: 6px; height: 14px; }
    footer { text-align: center; font-size: 12px; color: #64748B; padding: 8px 0 28px; }
  </style>
</head>
<body>
  <header>
    <h1>Análise de Reclamações de Clientes</h1>
    <p>Envie a planilha de reclamações ou cole um texto para priorizar as melhorias do produto.</p>
  </header>

  <main>
    <!-- 1. UPLOAD DA PLANILHA (topo da tela) -->
    <form method="post" action="/planilha" enctype="multipart/form-data"
          class="cartao upload" id="form-upload">
      <div>
        <h2>Planilha de reclamações</h2>
        <p class="dica">Formatos aceitos: .csv e .xlsx (até {{ limite_mb }} MB). Colunas esperadas:
          categoria/motivo, status e canal de atendimento.</p>
        <p id="estado-upload"></p>
      </div>
      <label for="arquivo" class="botao-upload">📁 Enviar planilha</label>
      <input type="file" id="arquivo" name="arquivo" accept=".csv,.xlsx">
    </form>

    {% if erro_planilha %}<div class="alerta">{{ erro_planilha }}</div>{% endif %}
    {% if aviso_planilha %}<div class="aviso">{{ aviso_planilha }}</div>{% endif %}

    <!-- 2. PAINEL COM OS GRÁFICOS DE PIZZA -->
    {% if painel %}
      <p class="resumo">Arquivo <b>{{ painel.arquivo }}</b> &middot;
        {{ painel.linhas }} reclamações analisadas</p>
      <div class="grade">
        {% for g in painel.graficos %}
          <div class="cartao">
            <h2>{{ g.titulo }}</h2>
            <p class="dica">Coluna usada: {{ g.coluna }}</p>
            <div class="area-grafico"><canvas id="grafico-{{ g.id }}"></canvas></div>
          </div>
        {% endfor %}
      </div>
      <div class="espaco"></div>
    {% endif %}

    <!-- 3. ANÁLISE DE TEXTO -->
    <form method="post" class="cartao">
      <h2 style="margin-bottom:12px;">Análise de texto</h2>
      <label for="texto">Texto da reclamação</label>
      <textarea id="texto" name="texto" maxlength="{{ limite }}"
                placeholder="Cole aqui uma ou mais reclamações...">{{ texto }}</textarea>

      <div class="linha">
        <span class="contador"><span id="qtd">0</span> / {{ limite }} caracteres</span>
        <div>
          <label for="top_n" style="display:inline; margin-right:6px;">Exibir</label>
          <select id="top_n" name="top_n">
            {% for op in opcoes %}
              <option value="{{ op }}" {% if op == top_n %}selected{% endif %}>
                {{ op }} palavras
              </option>
            {% endfor %}
          </select>
        </div>
      </div>

      <div class="linha">
        <button type="button" class="secundario" id="exemplo">Usar texto de exemplo</button>
        <button type="submit" class="principal">Analisar</button>
      </div>
    </form>

    {% if erro %}<div class="alerta">{{ erro }}</div>{% endif %}
    {% if aviso %}<div class="aviso">{{ aviso }}</div>{% endif %}

    {% if resultado and resultado.ranking %}
      <div class="metricas">
        <div class="metrica"><b>{{ resultado.total_palavras }}</b><span>Palavras no texto</span></div>
        <div class="metrica"><b>{{ resultado.total_analisadas }}</b><span>Palavras analisadas</span></div>
        <div class="metrica"><b>{{ resultado.total_diferentes }}</b><span>Palavras diferentes</span></div>
      </div>

      <div class="cartao">
        <h2 style="margin-bottom:12px;">Palavras mais frequentes</h2>
        <table>
          <thead>
            <tr><th>#</th><th>Palavra</th><th>Ocorrências</th><th>Participação</th><th style="width:35%;"></th></tr>
          </thead>
          <tbody>
            {% for item in resultado.ranking %}
              <tr>
                <td>{{ item.posicao }}</td>
                <td><b>{{ item.palavra }}</b></td>
                <td>{{ item.quantidade }}</td>
                <td>{{ item.participacao }}</td>
                <td><div class="barra-fundo"><div class="barra" style="width: {{ item.largura }}%"></div></div></td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    {% endif %}
  </main>

  <footer>Ferramenta interna. Os resultados apoiam a análise e não substituem a leitura das reclamações.</footer>

  {% if painel %}
    <!-- Biblioteca de gráficos (Chart.js), carregada da internet -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  {% endif %}

  <script>
    // --- Upload: ao escolher o arquivo, envia automaticamente ---
    const entradaArquivo = document.getElementById("arquivo");
    entradaArquivo.addEventListener("change", () => {
      if (entradaArquivo.files.length > 0) {
        document.getElementById("estado-upload").textContent =
          "Processando " + entradaArquivo.files[0].name + "...";
        document.getElementById("form-upload").submit();
      }
    });

    // --- Análise de texto: contador de caracteres e botão de exemplo ---
    const campo = document.getElementById("texto");
    const contador = document.getElementById("qtd");
    const atualizar = () => { contador.textContent = campo.value.length; };
    campo.addEventListener("input", atualizar);
    document.getElementById("exemplo").addEventListener("click", () => {
      campo.value = {{ exemplo|tojson }};
      atualizar();
    });
    atualizar();

    // --- Gráficos de pizza ---
    const GRAFICOS = {{ (painel.graficos if painel else [])|tojson }};

    if (GRAFICOS.length > 0) {
      if (typeof Chart === "undefined") {
        // A biblioteca não carregou (sem internet ou bloqueada)
        document.querySelectorAll(".area-grafico").forEach((area) => {
          area.innerHTML = '<p class="aviso">Não foi possível carregar os gráficos. ' +
                           'Verifique a conexão com a internet.</p>';
        });
      } else {
        GRAFICOS.forEach((g) => {
          new Chart(document.getElementById("grafico-" + g.id), {
            type: "pie",
            data: {
              labels: g.legendas,   // legenda já traz a porcentagem
              datasets: [{
                data: g.valores,
                backgroundColor: g.cores,
                borderColor: "#FFFFFF",
                borderWidth: 2,
              }],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  position: "bottom",
                  labels: { boxWidth: 14, padding: 14, font: { size: 13 } },
                },
                // Ao passar o mouse: nome da fatia, quantidade e porcentagem
                tooltip: {
                  callbacks: {
                    title: (itens) => g.nomes[itens[0].dataIndex],
                    label: (item) =>
                      " " + g.valores[item.dataIndex] + " reclamações (" +
                      g.percentuais[item.dataIndex] + "%)",
                  },
                },
              },
            },
          });
        });
      }
    }
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 5. ROTAS
# ---------------------------------------------------------------------------
def render_pagina(**contexto):
    """Monta a página com valores padrão; só passamos o que muda em cada rota."""
    valores = dict(
        texto="", erro=None, aviso=None, resultado=None, top_n=TOP_PADRAO,
        erro_planilha=None, aviso_planilha=None, painel=None,
    )
    valores.update(contexto)
    return render_template_string(
        PAGINA, opcoes=OPCOES_TOP, limite=LIMITE_CARACTERES,
        limite_mb=LIMITE_ARQUIVO_MB, exemplo=EXEMPLO, **valores,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    """Página inicial e análise de texto digitado."""
    texto, erro, aviso, resultado = "", None, None, None
    top_n = TOP_PADRAO

    if request.method == "POST":
        texto = request.form.get("texto", "")

        # Lê a quantidade escolhida; se vier algo inválido, usa o padrão
        try:
            top_n = int(request.form.get("top_n", TOP_PADRAO))
        except ValueError:
            top_n = TOP_PADRAO
        if top_n not in OPCOES_TOP:
            top_n = TOP_PADRAO

        # Validações da entrada
        if not texto.strip():
            erro = "Cole o texto da reclamação para realizar a análise."
        elif len(texto) > LIMITE_CARACTERES:
            erro = f"O texto excede o limite de {LIMITE_CARACTERES} caracteres."
        else:
            resultado = analisar_texto(texto, top_n)
            if not resultado["ranking"]:
                aviso = ("Não foram encontradas palavras relevantes. "
                         "Tente um texto mais longo.")

    return render_pagina(texto=texto, erro=erro, aviso=aviso,
                         resultado=resultado, top_n=top_n)


@app.route("/planilha", methods=["GET", "POST"])
def planilha():
    """Recebe a planilha, detecta as colunas e gera os dados dos gráficos."""
    if request.method == "GET":
        return redirect("/")  # abrir /planilha direto volta para a página inicial

    arquivo = request.files.get("arquivo")
    if arquivo is None or not arquivo.filename:
        return render_pagina(erro_planilha="Selecione um arquivo .csv ou .xlsx.")
    if not arquivo.filename.lower().endswith(EXTENSOES):
        return render_pagina(
            erro_planilha="Formato não suportado. Envie um arquivo .csv ou .xlsx.")

    try:
        df = ler_planilha(arquivo)
    except Exception:
        return render_pagina(erro_planilha=(
            "Não foi possível ler o arquivo. Verifique se ele não está "
            "corrompido e se a primeira linha contém os nomes das colunas."))

    if df.empty:
        return render_pagina(erro_planilha="A planilha não possui nenhuma linha de dados.")

    # Descobre quais colunas correspondem a categoria, status e canal
    colunas = detectar_colunas(df)
    nomes_colunas = ", ".join(str(c) for c in list(df.columns)[:15])
    if not colunas:
        return render_pagina(erro_planilha=(
            "Não encontramos colunas de categoria/motivo, status ou canal. "
            f"Colunas do arquivo: {nomes_colunas}."))

    # Monta um gráfico para cada coluna encontrada (na ordem da tela)
    graficos = []
    for chave in GRAFICOS:
        if chave in colunas:
            dados = montar_distribuicao(df[colunas[chave]], chave)
            dados["coluna"] = str(colunas[chave])
            graficos.append(dados)

    aviso = None
    faltando = [GRAFICOS[c]["titulo"] for c in GRAFICOS if c not in colunas]
    if faltando:
        aviso = (f"Não encontramos a coluna de: {', '.join(faltando)}. "
                 f"Colunas do arquivo: {nomes_colunas}.")

    painel = {"arquivo": arquivo.filename, "linhas": len(df), "graficos": graficos}
    return render_pagina(painel=painel, aviso_planilha=aviso)


@app.errorhandler(413)
def arquivo_grande(_erro):
    """Chamado pelo Flask quando o arquivo passa do limite de tamanho."""
    return render_pagina(
        erro_planilha=f"O arquivo excede o limite de {LIMITE_ARQUIVO_MB} MB."), 413


# ---------------------------------------------------------------------------
# 6. EXECUÇÃO LOCAL
#    No Render, quem inicia o app é o gunicorn (veja o Start Command).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    porta = int(os.environ.get("PORT", 5000))

    # Com debug=True o Flask inicia 2 processos (o 2º é o "reiniciador").
    # Abrimos o navegador só no 1º, para a página não abrir duas vezes.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        # Espera 1,5 s para o servidor ligar e só então abre a página
        threading.Timer(
            1.5, lambda: webbrowser.open(f"http://127.0.0.1:{porta}")
        ).start()

    app.run(debug=True, port=porta)