import json
import re
from html import unescape
from pathlib import Path

INPUT_PATH  = Path("noticias_brutas.json")
OUTPUT_PATH = Path("dados_limpos.json")


def remover_html(texto):
    texto = re.sub(r'<!--.*?-->', '', texto, flags=re.DOTALL)
    texto = re.sub(r'<[^>]+>', ' ', texto)
    return texto


def remover_metadados(texto):
    texto = re.sub(r'Publicado\s+em\s*:.*', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\b\d{2}/\d{2}/\d{4}\b', '', texto)
    texto = re.sub(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z)?\b', '', texto)
    texto = re.sub(r'\d{1,2}h\d{2}\s*\|\s*\w+', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'https?://\S+', '', texto)
    texto = re.sub(r'www\.\S+', '', texto)
    texto = re.sub(r'Fonte\s*:\s*\S+', '', texto, flags=re.IGNORECASE)
    return texto


def normalizar_espacos(texto):
    texto = re.sub(r'[ \t]+', ' ', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    linhas = [linha.strip() for linha in texto.split('\n')]
    texto = '\n'.join(linhas)
    return texto.strip()


def limpar_texto(texto):
    texto = unescape(texto)
    texto = remover_html(texto)
    texto = remover_metadados(texto)
    texto = normalizar_espacos(texto)
    return texto


def tem_conteudo(texto, minimo=80):
    chars_uteis = len(texto.replace('\n', '').replace(' ', ''))
    return chars_uteis >= minimo


def processar_noticias(noticias):
    limpas = []
    descartadas = []

    for noticia in noticias:
        texto_limpo = limpar_texto(noticia['texto'])

        if not tem_conteudo(texto_limpo):
            descartadas.append(noticia['id'])
            print(f"[DESCARTADO] id={noticia['id']} — '{noticia['titulo']}'")
            continue

        limpas.append({
            'id':     noticia['id'],
            'titulo': noticia['titulo'].strip(),
            'texto':  texto_limpo,
            'data':   noticia['data'],
            'fonte':  noticia['fonte'],
        })

    return limpas, descartadas


if __name__ == '__main__':
    with open(INPUT_PATH, encoding='utf-8') as f:
        noticias = json.load(f)

    limpas, descartadas = processar_noticias(noticias)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(limpas, f, ensure_ascii=False, indent=2)

    print(f"\nTotal bruto  : {len(noticias)}")
    print(f"Aproveitados : {len(limpas)}")
    print(f"Descartados  : {len(descartadas)} (ids: {descartadas})")
    print(f"Salvo em     : {OUTPUT_PATH}")
