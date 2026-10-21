# Dados do Sisu 2025

`sisu2025_ofertas_notas_corte.csv` é o arquivo `data/2025/scraping/grades.csv` do repositório
[maiakanegae/sisu-data](https://github.com/maiakanegae/sisu-data) (licença MIT), coletado da API
pública do Sisu/MEC. Dados não oficiais, usados aqui só para estudo.

Uma linha por oferta (curso + turno + campus + instituição). Sem cabeçalho, separador `;`:

| Posição | Conteúdo |
|---|---|
| 0–9 | código da oferta, curso, grau, turno, vagas totais, campus, município, UF, instituição, sigla |
| 10–14 | pesos de Ciências da Natureza, Ciências Humanas, Linguagens, Matemática e Redação |
| 15–19 | notas mínimas nas mesmas provas (0,01 = só exige nota maior que zero) |
| 20 | média mínima (média simples das 5 provas) |
| 21 em diante | blocos de 5 campos por modalidade: nome, vagas, nota de corte, % de bônus, data da nota de corte |

As notas de corte são as do fim das inscrições do Sisu 2025 (22/01/2025), que usou as notas do ENEM 2024.
