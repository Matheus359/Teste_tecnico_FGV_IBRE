## O que esse projeto faz

Você tem 20 notícias econômicas salvas em um arquivo JSON. O problema é que esses textos estão cheios de sujeira: tags HTML, datas embutidas, URLs e outros ruídos que atrapalham a leitura.

Esse projeto resolve isso em três passos:

1. **Limpa** os textos, removendo toda a sujeira
2. **Transforma** cada notícia em um vetor numérico que representa o seu significado
3. **Busca** as notícias mais relevantes para uma pergunta feita em português

## Como rodar

### 1. Instalar as dependências

pip install -r requirements.txt

### 2. Rodar as etapas na ordem

python etapa1_limpeza.py
python etapa2_embeddings.py
python etapa3_busca.py

### A etapa 3 roda três buscas de validação automaticamente e depois entra em modo interativo, onde você pode digitar qualquer busca. Para sair, aperte Enter em branco.

---

## Arquivos do projeto

```
├── noticias_brutas.json      # Arquivo original sem o tratamento.
├── dados_limpos.json         # Arquivo gerado pela etapa 1
├── etapa1_limpeza.py
├── etapa2_embeddings.py
├── etapa3_busca.py
└── requirements.txt          # requerimentos necessários para executar o teste.
```

Os arquivos `embeddings.npy`, `embeddings_meta.pkl` e `tfidf_pipeline.pkl` são gerados automaticamente pela etapa 2.


## Etapa 1 — Limpeza

O campo `texto` de cada notícia vinha com vários problemas. A limpeza é feita em quatro funções:

- `remover_html` — tira tags como `<p>`, `<strong>`, `<a href="...">` e comentários `<!-- -->`
- `remover_metadados` — tira datas, horários, URLs e cabeçalhos como "Publicado em:"
- `normalizar_espacos` — arruma espaços e quebras de linha em excesso
- `tem_conteudo` — descarta notícias com menos de 80 caracteres úteis após a limpeza

Exemplo: A notícia de id=18 ("Nota curta") foi descartada pois após a limpeza sobrou apenas a palavra "Selic".


## Etapa 2 — Embeddings

Cada notícia é convertida em uma lista de números (vetor) que representa o seu significado. Notícias com assuntos parecidos ficam com vetores parecidos — e é isso que permite a busca semântica funcionar.

O modelo ideal para isso seria o **BERTimbau** (`neuralmind/bert-large-portuguese-cased`), treinado especificamente em textos em português. Ele foi escolhido por entender melhor termos econômicos como "Copom", "Selic" e "arcabouço fiscal" do que modelos genéricos.

Como o modelo precisa ser baixado da internet, o código detecta automaticamente se ele está disponível. Se não estiver, usa um método alternativo chamado **TF-IDF + LSA**, que funciona offline e já vem instalado com o scikit-learn.


## Etapa 3 — Busca

Quando você digita uma busca, o sistema:

1. Transforma a sua pergunta em um vetor, usando o mesmo método da etapa 2
2. Compara esse vetor com o vetor de cada notícia
3. Retorna as 3 notícias com maior similaridade, junto com o score e um trecho relevante

## Resultados

### "mudancas na taxa de juros"
| # | Score | Notícia |
|---|---|---|
| 1 | 70% | Taxa de desemprego cai para 7,9% (id=4) |
| 2 | 47% | Desemprego juvenil ainda preocupa (id=14) |
| 3 | 44% | Dólar fecha abaixo de R$ 4,80 (id=5) |

### "mercado de trabalho e desemprego"
| # | Score | Notícia |
|---|---|---|
| 1 | 82% | Desemprego juvenil ainda preocupa (id=14) |
| 2 | 48% | Setor de serviços cresce 0,6% (id=19) |
| 3 | 45% | Câmbio: real se fortalece (id=15) |

### "inflacao e precos ao consumidor"
| # | Score | Notícia |
|---|---|---|
| 1 | 76% | IPA desacelera, pressão sobre preços diminui (id=9) |
| 2 | 59% | IPCA de julho registra 0,12% (id=2) |
| 3 | 41% | Confiança do consumidor sobe (id=17) |