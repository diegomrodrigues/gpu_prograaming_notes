import os
from dotenv import load_dotenv

load_dotenv()

from pollo.agents.draft.writer import generate_drafts_from_topics

drafts = generate_drafts_from_topics(
    directory="10. Compute Clusters",
    perspectives=[
        "Foque nos fundamentos teóricos da programação CUDA e GPU, incluindo arquitetura de hardware, modelos de memória, execução paralela e otimização de algoritmos. Explore os aspectos matemáticos subjacentes à computação paralela e as estruturas formais para análise de desempenho.",
        "Foque nos algoritmos e técnicas fundamentais para programação GPU, incluindo paralelismo de dados, padrões de acesso à memória, sincronização e modelos de programação como CUDA, OpenCL e bibliotecas de alto nível. Aborde também técnicas de otimização como coalescing, tiling, e redução de divergência de warp.",
    ],
    json_per_perspective=1,
    branching_factor=5
)