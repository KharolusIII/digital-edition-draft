
# Digital Edition Draft

This repository runs a pipeline that transforms a small corpus of Latin poems into progressively enriched TEI (entities, topics, validation, visualizations). Each stage is a script you can re-run independently.

---

## English

### Overview
- Texts live under `data_root/<author>/` (e.g., `Catulo/`, `Tibulo/`, `Propercio/`). If you start from `data_root_export/`, rename it to `data_root/` first.
- File naming convention for poems: `Author_Poem_Number.txt` (e.g., `Catulo_Carmen_005.txt`).
- Numbered scripts (`0_*.py` … `8_*.py`) consume those texts and generate TEI in `data_root/tei_pipeline/<step>/`, ending with Relax NG-validated TEI in `08_tei_header_with_relaxng/{standoff,flatten,combined}`.
- Entity/topic CSVs and linkouts live in `data_root/entities/` and `data_root/outputs/`.

### Requirements
- Python 3.10+ (`py -3.11` on Windows).
- Install deps: `pip install -r requirements.txt`.
- LatinCy model (not tracked):  
  `py -3 -m pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl`

### Running the pipeline
1) Ensure `data_root/` exists (rename from `data_root_export/` if needed) and contains the per-author folders plus optional PDFs (Index Nominum) and Pleiades CSVs.
2) Run scripts in order; each can be rerun alone:  
   `0` TEI skeletons + verse count → `02/03`  
   `1` NER into TEI + entity CSVs/index checks → `04`  
   `2` VIAF, `3` Pleiades, `4` Wikidata (may hit 403) → enriched entities  
   `5` Topic matching (Soldevila) → CSVs per author (`data_root/topics/`)  
   `6` Topic annotation → `05` standoff, `06` flatten, `07` combined  
   `7` XML-model + Relax NG validation → `08_tei_header_with_relaxng/{standoff,flatten,combined}`
3) Final TEI are under `data_root/tei_pipeline/08_tei_header_with_relaxng/`.

### Soldevila dictionary
- Reference: Moreno Soldevila, R. (2011). *Diccionario de motivos amatorios en la literatura latina*. Editorial Universidad de Huelva. http://rabida.uhu.es/dspace/handle/10272/14398
- For space/time we ship only a tiny simulated sample; to run the full topic matcher, download the dictionary from the link and place it at `data_root/soldevila/Soldevila8.txt` (or adjust the path in `5_topic_matching.py`).

---

## Español

### Descripción general
- Los textos se encuentran en `data_root/<autor>/` (p.ej., `Catulo/`, `Tibulo/`, `Propercio/`). Si partes de `data_root_export/`, renómbralo a `data_root/` antes de ejecutar.
- Convención de nombres para poemas: `Autor_Poema_Numero.txt` (ej.: `Catulo_Carmen_005.txt`).
- Los scripts numerados (`0_` … `8_`) procesan esos textos y guardan TEI en `data_root/tei_pipeline/<etapa>/`, con los TEI validados en `08_tei_header_with_relaxng/{standoff,flatten,combined}`.
- Los CSV de entidades y tópicos están en `data_root/entities/` y `data_root/outputs/`.

### Requisitos
- Python 3.10+ (`py -3.11` en Windows).
- Instala dependencias: `pip install -r requirements.txt`.
- Modelo LatinCy (no incluido):  
  `py -3 -m pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl`

### Ejecución
1) Asegura `data_root/` (renombrado desde `data_root_export/` si hace falta) con las carpetas por autor y, opcionalmente, PDFs de índices y CSV de Pleiades.
2) Corre los scripts en orden:  
   `0` TEI base + conteo de versos → `02/03`  
   `1` NER + CSV/índices → `04`  
   `2` VIAF, `3` Pleiades, `4` Wikidata (puede dar 403) → entidades enriquecidas  
   `5` Tópicos (Soldevila) → CSV por autor (`data_root/topics/`)  
   `6` Tópicos en TEI → `05` standoff, `06` flatten, `07` combinado  
   `7` Cabecera xml-model + validación Relax NG → `08_tei_header_with_relaxng/{standoff,flatten,combined}`
3) Revisa los TEI finales en `data_root/tei_pipeline/08_tei_header_with_relaxng/`.

### Diccionario de Soldevila
- Referencia: Moreno Soldevila, R. (2011). *Diccionario de motivos amatorios en la literatura latina*. Editorial Universidad de Huelva. http://rabida.uhu.es/dspace/handle/10272/14398
- Por tiempo y espacio usamos una simulación mínima; para el matcher completo, descarga el diccionario y guárdalo en `data_root/soldevila/Soldevila8.txt` (o ajusta la ruta en `5_topic_matching.py`).

---

## Português

### Visão geral
- Os textos ficam em `data_root/<autor>/` (ex.: `Catulo/`, `Tibulo/`, `Propercio/`). Se você partir de `data_root_export/`, renomeie para `data_root/` antes de rodar.
- Convenção de nomes dos poemas: `Autor_Poema_Numero.txt` (ex.: `Catulo_Carmen_005.txt`).
- Os scripts `0_` … `8_` geram TEI em `data_root/tei_pipeline/<etapa>/`, com TEI validados em `08_tei_header_with_relaxng/{standoff,flatten,combined}`.
- CSVs de entidades e tópicos ficam em `data_root/entities/` e `data_root/outputs/`.

### Requisitos
- Python 3.10+.
- `pip install -r requirements.txt`.
- Modelo LatinCy (não incluído):  
  `py -3 -m pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl`

### Como executar
1) Garanta `data_root/` (renomeado de `data_root_export/` se preciso) com as pastas por autor e, opcionalmente, PDFs de índices e CSV de Pleiades.
2) Rode em ordem:  
   `0` TEI base + contagem de versos → `02/03`  
   `1` NER + CSV/índices → `04`  
   `2` VIAF, `3` Pleiades, `4` Wikidata (pode dar 403) → entidades enriquecidas  
   `5` Tópicos (Soldevila) → CSV por autor (`data_root/topics/`)  
   `6` Tópicos em TEI → `05` standoff, `06` flatten, `07` combinado  
   `7` xml-model + validação Relax NG → `08_tei_header_with_relaxng/{standoff,flatten,combined}`
3) TEI finais em `data_root/tei_pipeline/08_tei_header_with_relaxng/`.

### Dicionário de Soldevila
- Referência: Moreno Soldevila, R. (2011). *Diccionario de motivos amatorios en la literatura latina*. Editorial Universidad de Huelva. http://rabida.uhu.es/dspace/handle/10272/14398
- Aqui usamos uma amostra simulada por limitação de tempo/tamanho; para o matcher completo, baixe o dicionário e coloque em `data_root/soldevila/Soldevila8.txt` (ou ajuste a rota em `5_topic_matching.py`).

---

## data_root_export
- `texts/`: sample poems and a tiny simulated dictionary; rename the whole folder to `data_root/` before running the pipeline, and delete anything you do not wish to share.
- `pleiades/`: Pleiades CSVs for place linking.
- TEI outputs are not shipped here to keep the bundle light; the pipeline regenerates them.
