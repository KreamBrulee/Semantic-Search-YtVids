from txtai.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

df = open('transcriptses.csv', 'r')

sentences = []

i = 0
for elim in df:
    if i != 0:
        sentences.append(elim[0])
    print(i)
    i += 1

embendings=model.encode(sentences)

for sen,emb in zip(sentences,embendings):
    print(sen+'::'+emb)