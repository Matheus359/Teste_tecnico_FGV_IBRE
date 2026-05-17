import json
import numpy as np
import pickle
from pathlib import Path

INPUT_PATH  = Path("dados_limpos.json")
OUTPUT_NPY  = Path("embeddings.npy")
OUTPUT_META = Path("embeddings_meta.pkl")

MODELO_ST = "neuralmind/bert-large-portuguese-cased"
LSA_DIM   = 128


def preparar_texto(noticia):
    return f"{noticia['titulo']}. {noticia['titulo']}. {noticia['texto']}"


def gerar_com_sentence_transformers(textos):
    from sentence_transformers import SentenceTransformer

    print(f"Carregando modelo: {MODELO_ST}")
    modelo = SentenceTransformer(MODELO_ST)

    print(f"Codificando {len(textos)} documentos...")
    embeddings = modelo.encode(textos, batch_size=8, show_progress_bar=True, normalize_embeddings=True)

    return embeddings, MODELO_ST


def gerar_com_tfidf(textos):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    print("[FALLBACK] sentence-transformers indisponivel. Usando TF-IDF + LSA.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)
    matriz_tfidf = vectorizer.fit_transform(textos)

    n_components = min(LSA_DIM, matriz_tfidf.shape[1] - 1, matriz_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(matriz_tfidf)
    embeddings = normalize(embeddings, norm='l2')

    pipeline_path = Path("tfidf_pipeline.pkl")
    with open(pipeline_path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'svd': svd}, f)
    print(f"Pipeline TF-IDF salvo em: {pipeline_path}")

    nome_modelo = f"TF-IDF(1,2)-LSA-{n_components}d [fallback]"
    return embeddings, nome_modelo


def salvar_embeddings(embeddings, ids, modelo_usado):
    np.save(OUTPUT_NPY, embeddings)

    meta = {
        'ids':      ids,
        'modelo':   modelo_usado,
        'dimensao': embeddings.shape[1],
        'n_docs':   embeddings.shape[0],
    }
    with open(OUTPUT_META, 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nEmbeddings gerados.")
    print(f"  Modelo   : {modelo_usado}")
    print(f"  Shape    : {embeddings.shape}  (docs x dimensao)")
    print(f"  Salvo em : {OUTPUT_NPY}")


if __name__ == '__main__':
    with open(INPUT_PATH, encoding='utf-8') as f:
        noticias = json.load(f)

    ids    = [n['id'] for n in noticias]
    textos = [preparar_texto(n) for n in noticias]

    print(f"Documentos a codificar: {len(textos)}")

    try:
        import sentence_transformers
        embeddings, modelo_usado = gerar_com_sentence_transformers(textos)
    except Exception as e:
        print(f"Aviso: {e}")
        embeddings, modelo_usado = gerar_com_tfidf(textos)

    salvar_embeddings(embeddings, ids, modelo_usado)