
# Digital Edition Draft

This repository runs a pipeline that transforms a small corpus of Latin poems into progressively enriched TEI (entities, topics, validation, visualizations). Each stage is a script you can re-run independently.

---

## English

### Overview
- Texts live under `data_root/<author>/` (e.g., `Catulo/`, `Tibulo/`, `Propercio/`). If you start from `data_root_export/`, rename it to `data_root/` first.
- File naming convention for poems: `Author_Poem_Number.txt` (e.g., `Catulo_Carmen_005.txt`).
- Numbered scripts (`0_*.py` ... `8_*.py`) consume those texts and generate TEI in `data_root/tei_pipeline/<step>/`, ending with Relax NG-validated TEI in `08_tei_header_with_relaxng/{standoff,flatten,combined}`. The TEI header declares the Soldevila taxonomy; categories are added only for topics actually detected.
- Entity/topic CSVs and linkouts live in `data_root/entities/` and `data_root/outputs/`.

### Requirements
- Python 3.10+ (`py -3.11` on Windows).
- Install deps: `pip install -r requirements.txt`.
- LatinCy model (not tracked):  
  `py -3 -m pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl`

### Running the pipeline
1) Ensure `data_root/` exists (rename from `data_root_export/` if needed) and contains the per-author folders plus optional PDFs (Index Nominum) and Pleiades CSVs.
2) Run scripts in order:  
   `0` TEI skeletons + verse count (declares Soldevila taxonomy in the header) -> `02/03`  
   `1` NER into TEI + entity CSVs/index checks -> `04`  
   `2` VIAF, `3` Pleiades, `4` Wikidata (may hit 403) -> enriched entities  
   `5` Topic matching (Soldevila) -> CSVs per author (`data_root/topics/`)  
   `6` Topic annotation (+ Soldevila categories only for detected topics) -> `05` standoff, `06` flatten, `07` combined  
   `7` XML-model + Relax NG validation -> `08_tei_header_with_relaxng/{standoff,flatten,combined}`
3) Final TEI are under `data_root/tei_pipeline/08_tei_header_with_relaxng/`.

### Soldevila dictionary
- Reference: Moreno Soldevila, R. (Ed.). (2011). *Diccionario de motivos amatorios en la Literatura Latina*. Universidad de Huelva. http://hdl.handle.net/10272/14398
- For space/time we ship only a tiny simulated sample; to run the full topic matcher, download the dictionary from the link and place it at `data_root/soldevila/Soldevila8.txt` (or adjust the path in `5_topic_matching.py`).

## What is automatic, semi-automatic, and manual in this pipeline?

| File | Level | What it does | Output / evidence | Human action |
|------|-------|--------------|-------------------|--------------|
| `0_tei_header_and_verses.py` | Automatic | Creates TEI base from TXT, builds `<teiHeader>`, splits verses into `<lg>/<l>`, declares Soldevila taxonomy. | `data_root/tei_pipeline/02_tei_header_generated`, `03_tei_verses_counted` | Review only if structure or metadata looks inconsistent. |
| `1_entities_pipeline.py` | Semi-automatic | Runs NER (LatinCy), tags TEI with PERSON/LOC/NORP, lemmatizes and checks indices. | `data_root/entities/*.csv`, `data_root/tei_pipeline/04_tei_entities_tagged` | Validate entity detection, types, and false positives/negatives. |
| `2_viaf_linking.py` | Semi-automatic | Queries VIAF and stores candidate URIs. | `data_root/entities/entities_index_checked_viaf.csv`, `data_root/entities/viaf_json/` | Choose the correct authority URI, discard bad candidates. |
| `3_pleiades_linking.py` | Semi-automatic | Adds Pleiades candidates and stats for places. | `data_root/entities/entities_index_checked_viaf_pleiades.csv`, `data_root/entities/pleiades_stats_*.csv` | Confirm correct gazetteer matches. |
| `4_wikidata_linking.py` | Semi-automatic | Queries Wikidata and adds candidate URIs. | `data_root/entities/entities_enriched_wikidata.csv` | Select final Wikidata IDs and resolve ambiguity. |
| `5_topic_matching.py` | Semi-automatic | Topic matching with Soldevila dictionary (n-grams + Levenshtein). | `data_root/topics/*_matches*.csv`, `data_root/soldevila/soldevila_ngrams.json` | Review topic matches; remove false positives. |
| `6_tei_topic_annotation.py` | Automatic (application) | Writes topic annotations (stand-off / inline / combined) and Soldevila categories for detected topics. | `data_root/tei_pipeline/05_topics_standoff`, `06_topics_flatten`, `07_topics_combined` | Spot-check annotation placement and consistency. |
| `7_tei_relaxng_header_and_validation.py` | Automatic | Inserts `xml-model`, enforces TEI namespace, validates against Relax NG. | `data_root/tei_pipeline/08_tei_header_with_relaxng/*`, validation logs | Address validation errors if any. |
| `8_tei_visualizations.py` | Automatic (output) | Generates plots and CSV summaries from TEI and CSV data. | `data_root/tei_pipeline/09_tei_visualizations/` | Interpret results; decide what to publish. |
| (Editorial decisions outside the scripts) | Manual | Final entity linking, disambiguation, taxonomy adjustments, accept/reject annotations. | Curated TEI and decision logs | Perform final scholarly validation. |

**Project principle:** repetitive tasks are automated, while final scholarly decisions remain under human editorial supervision.

---

## Español

### Descripción general
- Los textos se encuentran en `data_root/<autor>/` (p.ej., `Catulo/`, `Tibulo/`, `Propercio/`). Si partes de `data_root_export/`, renómbralo a `data_root/` antes de ejecutar.
- Convención de nombres para poemas: `Autor_Poema_Numero.txt` (ej.: `Catulo_Carmen_005.txt`).
- Los scripts numerados (`0_` ... `8_`) procesan esos textos y guardan TEI en `data_root/tei_pipeline/<etapa>/`, con los TEI validados en `08_tei_header_with_relaxng/{standoff,flatten,combined}`. El encabezado TEI declara la taxonomía Soldevila; las categorías se añaden solo para los tópicos detectados.
- Los CSV de entidades y tópicos están en `data_root/entities/` y `data_root/outputs/`.

### Requisitos
- Python 3.10+ (`py -3.11` en Windows).
- Instala dependencias: `pip install -r requirements.txt`.
- Modelo LatinCy (no incluido):  
  `py -3 -m pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl`

### Ejecución
1) Asegura `data_root/` (renombrado desde `data_root_export/` si hace falta) con las carpetas por autor y, opcionalmente, PDFs de índices y CSV de Pleiades.
2) Corre los scripts en orden:  
   `0` TEI base + conteo de versos (declara la taxonomia Soldevila en el header) -> `02/03`  
   `1` NER + CSV/índices → `04`  
   `2` VIAF, `3` Pleiades, `4` Wikidata (puede dar 403) -> entidades enriquecidas  
   `5` Tópicos (Soldevila) → CSV por autor (`data_root/topics/`)  
   `6` Tópicos en TEI → `05` standoff, `06` flatten, `07` combinado  
   `7` Cabecera xml-model + validación Relax NG → `08_tei_header_with_relaxng/{standoff,flatten,combined}`
3) Revisa los TEI finales en `data_root/tei_pipeline/08_tei_header_with_relaxng/`.

### Diccionario de Soldevila
- Referencia: Moreno Soldevila, R. (Ed.). (2011). *Diccionario de motivos amatorios en la Literatura Latina*. Universidad de Huelva. http://hdl.handle.net/10272/14398
- Por tiempo y espacio usamos una simulación mínima; para el matcher completo, descarga el diccionario y guárdalo en `data_root/soldevila/Soldevila8.txt` (o ajusta la ruta en `5_topic_matching.py`).

## ¿Qué es automático, semiautomático y manual en este pipeline?

| Archivo | Nivel | ¿Qué hace | Salida / evidencia | ¿Qué hace el editor humano |
|---------|-------|---------|--------------------|---------------------------|
| `0_tei_header_and_verses.py` | Automático | Crea TEI base desde TXT, arma `<teiHeader>`, divide versos en `<lg>/<l>`, declara la taxonomía Soldevila. | `data_root/tei_pipeline/02_tei_header_generated`, `03_tei_verses_counted` | Revisa solo si la estructura o metadatos son inconsistentes. |
| `1_entities_pipeline.py` | Semiautomático | Ejecuta NER (LatinCy), marca TEI con PERSON/LOC/NORP, lematiza y cruza índices. | `data_root/entities/*.csv`, `data_root/tei_pipeline/04_tei_entities_tagged` | Valida detecciones, tipos y falsos positivos/negativos. |
| `2_viaf_linking.py` | Semiautomático | Consulta VIAF y guarda candidatos/URIs. | `data_root/entities/entities_index_checked_viaf.csv`, `data_root/entities/viaf_json/` | Selecciona el URI correcto y descarta candidatos erróneos. |
| `3_pleiades_linking.py` | Semiautomático | Agrega candidatos de Pleiades y estadísticas. | `data_root/entities/entities_index_checked_viaf_pleiades.csv`, `data_root/entities/pleiades_stats_*.csv` | Confirma coincidencias toponímicas. |
| `4_wikidata_linking.py` | Semiautomático | Consulta Wikidata y agrega candidatos/URIs. | `data_root/entities/entities_enriched_wikidata.csv` | Elige IDs definitivos y resuelve ambigüedades. |
| `5_topic_matching.py` | Semiautomático | Matching de tópicos con Soldevila (n-gramas + Levenshtein). | `data_root/topics/*_matches*.csv`, `data_root/soldevila/soldevila_ngrams.json` | Revisa tópicos y elimina falsos positivos. |
| `6_tei_topic_annotation.py` | Automático (aplicación) | Escribe anotaciones temáticas (stand-off / inline / combined) y categorías Soldevila para tópicos detectados. | `data_root/tei_pipeline/05_topics_standoff`, `06_topics_flatten`, `07_topics_combined` | Controla consistencia y ubicación de las anotaciones. |
| `7_tei_relaxng_header_and_validation.py` | Automático | Inserta `xml-model`, asegura namespace TEI y valida Relax NG. | `data_root/tei_pipeline/08_tei_header_with_relaxng/*`, logs de validación | Corrige errores si aparecen. |
| `8_tei_visualizations.py` | Automático (salida) | Genera gráficos y CSVs a partir de TEI y datos. | `data_root/tei_pipeline/09_tei_visualizations/` | Interpreta resultados y decide qué publicar. |
| (Decisiones editoriales fuera de los scripts) | Manual | Entity linking final, desambiguaciones, ajustes taxonómicos, aceptar/rechazar anotaciones. | TEI curado y registro de decisiones | Validación filológica final. |

**Principio del proyecto:** se automatizan tareas repetitivas, pero la decisión final queda bajo supervisión editorial humana.

---

## Português

### Visão geral
- Os textos ficam em `data_root/<autor>/` (ex.: `Catulo/`, `Tibulo/`, `Propercio/`). Se você partir de `data_root_export/`, renomeie para `data_root/` antes de rodar.
- Convenção de nomes dos poemas: `Autor_Poema_Numero.txt` (ex.: `Catulo_Carmen_005.txt`).
- Os scripts `0_` ... `8_` geram TEI em `data_root/tei_pipeline/<etapa>/`, com TEI validados em `08_tei_header_with_relaxng/{standoff,flatten,combined}`. O header TEI declara a taxonomia Soldevila; as categorias só entram quando os tópicos aparecem.
- CSVs de entidades e tópicos ficam em `data_root/entities/` e `data_root/outputs/`.

### Requisitos
- Python 3.10+.
- `pip install -r requirements.txt`.
- Modelo LatinCy (não incluído):  
  `py -3 -m pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl`

### Como executar
1) Garanta `data_root/` (renomeado de `data_root_export/` se preciso) com as pastas por autor e, opcionalmente, PDFs de índices e CSV de Pleiades.
2) Rode em ordem:  
   `0` TEI base + contagem de versos (declara a taxonomia Soldevila no header) -> `02/03`  
   `1` NER + CSV/índices → `04`  
   `2` VIAF, `3` Pleiades, `4` Wikidata (pode dar 403) -> entidades enriquecidas  
   `5` Tópicos (Soldevila) → CSV por autor (`data_root/topics/`)  
   `6` Tópicos em TEI → `05` standoff, `06` flatten, `07` combinado  
   `7` xml-model + validação Relax NG → `08_tei_header_with_relaxng/{standoff,flatten,combined}`
3) TEI finais em `data_root/tei_pipeline/08_tei_header_with_relaxng/`.

### Dicionário de Soldevila
- Referência: Moreno Soldevila, R. (Ed.). (2011). *Diccionario de motivos amatorios en la Literatura Latina*. Universidad de Huelva. http://hdl.handle.net/10272/14398
- Aqui usamos uma amostra simulada por limitação de tempo/tamanho; para o matcher completo, baixe o dicionário e coloque em `data_root/soldevila/Soldevila8.txt` (ou ajuste a rota em `5_topic_matching.py`).

## O que é automático, semiautomático e manual neste pipeline?

| Arquivo | Nível | O que faz | Saída / evidência | O que o editor humano faz |
|---------|-------|----------|-------------------|---------------------------|
| `0_tei_header_and_verses.py` | Automático | Cria TEI base a partir de TXT, monta `<teiHeader>`, divide versos em `<lg>/<l>`, declara a taxonomia Soldevila. | `data_root/tei_pipeline/02_tei_header_generated`, `03_tei_verses_counted` | Revisar apenas se houver inconsistências estruturais/metadados. |
| `1_entities_pipeline.py` | Semiautomático | Executa NER (LatinCy), marca TEI com PERSON/LOC/NORP, lematiza e cruza índices. | `data_root/entities/*.csv`, `data_root/tei_pipeline/04_tei_entities_tagged` | Validar detecções, tipos e falsos positivos/negativos. |
| `2_viaf_linking.py` | Semiautomático | Consulta VIAF e salva candidatos/URIs. | `data_root/entities/entities_index_checked_viaf.csv`, `data_root/entities/viaf_json/` | Selecionar o URI correto e descartar candidatos errados. |
| `3_pleiades_linking.py` | Semiautomático | Adiciona candidatos de Pleiades e estatísticas. | `data_root/entities/entities_index_checked_viaf_pleiades.csv`, `data_root/entities/pleiades_stats_*.csv` | Confirmar correspondências toponímicas. |
| `4_wikidata_linking.py` | Semiautomático | Consulta Wikidata e adiciona candidatos/URIs. | `data_root/entities/entities_enriched_wikidata.csv` | Escolher IDs finais e resolver ambiguidades. |
| `5_topic_matching.py` | Semiautomático | Matching de tópicos com Soldevila (n-gramas + Levenshtein). | `data_root/topics/*_matches*.csv`, `data_root/soldevila/soldevila_ngrams.json` | Revisar tópicos e eliminar falsos positivos. |
| `6_tei_topic_annotation.py` | Automático (aplicação) | Escreve anotações temáticas (stand-off / inline / combined) e categorias Soldevila para tópicos detectados. | `data_root/tei_pipeline/05_topics_standoff`, `06_topics_flatten`, `07_topics_combined` | Checar consistência e posicionamento das anotações. |
| `7_tei_relaxng_header_and_validation.py` | Automático | Insere `xml-model`, garante namespace TEI e valida Relax NG. | `data_root/tei_pipeline/08_tei_header_with_relaxng/*`, logs de validação | Corrigir erros se aparecerem. |
| `8_tei_visualizations.py` | Automático (saída) | Gera gráficos e CSVs a partir de TEI e dados. | `data_root/tei_pipeline/09_tei_visualizations/` | Interpretar resultados e decidir o que publicar. |
| (Decisões editoriais fora dos scripts) | Manual | Entity linking final, desambiguação, ajustes taxonómicos, aceitar/rejeitar anotações. | TEI curado e registro de decisões | Validação filológica final. |

**Princípio do projeto:** tarefas repetitivas são automatizadas, mas a decisão final permanece sob supervisão editorial humana.

---

## data_root_export
- `texts/`: sample poems and a tiny simulated dictionary; rename the whole folder to `data_root/` before running the pipeline, and delete anything you do not wish to share.
- `pleiades/`: Pleiades CSVs for place linking.
- TEI outputs are not shipped here to keep the bundle light; the pipeline regenerates them.