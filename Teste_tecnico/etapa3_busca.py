import json
import pickle
import textwrap
import numpy as np
from pathlib import Path

CLEAN_PATH = Path("dados_limpos.json")
EMB_NPY    = Path("embeddings.npy")
EMB_META   = Path("embeddings_meta.pkl")
PIPELINE_P = Path("tfidf_pipeline.pkl")

QUERIES_VALIDACAO = [
    "mudancas na taxa de juros",
    "mercado de trabalho e desemprego",
    "inflacao e precos ao consumidor",
]


def carregar_dados():
    with open(CLEAN_PATH, encoding='utf-8') as f:
        noticias = json.load(f)

    embeddings = np.load(EMB_NPY)

    with open(EMB_META, 'rb') as f:
        meta = pickle.load(f)

    tfidf_pipeline = None
    if 'fallback' in meta['modelo'] and PIPELINE_P.exists():
        with open(PIPELINE_P, 'rb') as f:
            tfidf_pipeline = pickle.load(f)

    print(f"Dados carregados | docs: {len(noticias)} | modelo: {meta['modelo']}")
    return noticias, embeddings, meta, tfidf_pipeline


def vetorizar_query(query, meta, tfidf_pipeline):
    if 'fallback' not in meta['modelo']:
        from sentence_transformers import SentenceTransformer
        modelo = SentenceTransformer(meta['modelo'])
        vetor = modelo.encode([query], normalize_embeddings=True)
        return vetor[0]
    else:
        from sklearn.preprocessing import normalize
        tfidf_q = tfidf_pipeline['vectorizer'].transform([query])
        lsa_q   = tfidf_pipeline['svd'].transform(tfidf_q)
        return normalize(lsa_q, norm='l2')[0]


def extrair_snippet(texto, query, tamanho=200):
    termos    = set(query.lower().split())
    sentencas = [s.strip() for s in texto.replace('\n', ' ').split('.') if s.strip()]

    melhor       = sentencas[0] if sentencas else texto
    melhor_score = 0

    for sentenca in sentencas:
        palavras = set(sentenca.lower().split())
        score    = len(termos & palavras)
        if score > melhor_score:
            melhor       = sentenca
            melhor_score = score

    if len(melhor) > tamanho:
        melhor = melhor[:tamanho].rsplit(' ', 1)[0] + '...'

    return melhor

def buscar(query, noticias, embeddings, meta, tfidf_pipeline, top_k=3):
    id_para_noticia = {n['id']: n for n in noticias}

    vetor_query = vetorizar_query(query, meta, tfidf_pipeline)
    scores      = embeddings @ vetor_query
    indices     = np.argsort(scores)[::-1][:top_k]

    resultados = []
    for rank, idx in enumerate(indices, 1):
        doc_id  = meta['ids'][idx]
        noticia = id_para_noticia[doc_id]
        snippet = extrair_snippet(noticia['texto'], query)

        resultados.append({
            'rank':    rank,
            'id':      doc_id,
            'score':   float(scores[idx]),
            'titulo':  noticia['titulo'],
            'data':    noticia['data'],
            'fonte':   noticia['fonte'],
            'snippet': snippet,
        })

    return resultados

def exibir_resultados(query, resultados):
    print("\n" + "=" * 68)
    print(f'  QUERY: "{query}"')
    print("=" * 68)

    for r in resultados:
        barra = "=" * int(r['score'] * 20)
        print(f"\n  #{r['rank']}  [{r['score']:.1%}] {barra}")
        print(f"  {r['titulo']}")
        print(f"  {r['data']} | {r['fonte']} | id={r['id']}")
        print()
        snippet_fmt = textwrap.fill(r['snippet'], width=64,
                            initial_indent="  > ",
                            subsequent_indent="    ")
        print(snippet_fmt)

    print("\n" + "-" * 68)

if __name__ == '__main__':
    noticias, embeddings, meta, tfidf_pipeline = carregar_dados()

    for query in QUERIES_VALIDACAO:
        resultados = buscar(query, noticias, embeddings, meta, tfidf_pipeline)
        exibir_resultados(query, resultados)

    print("\nModo interativo (Enter em branco para sair):")
    while True:
        query = input("\n  Sua busca: ").strip()
        if not query:
            break
        resultados = buscar(query, noticias, embeddings, meta, tfidf_pipeline)
        exibir_resultados(query, resultados)