8 ## English
9 
10 ### Overview
11 - Texts live under `data_root/<author>/` (e.g., `Catulo/`, `Tibulo/`, `Propercio/`). If you start from `data_root_export/`, rename it to `data_root/` first.
12 - File naming convention for poems: `Author_Poem_Number.txt` (e.g., `Catulo_Carmen_005.txt`).
13 - Numbered scripts (`0_*.py` … `8_*.py`) consume those texts and generate TEI in `data_root/tei_pipeline/<step>/`, ending with Relax NG-validated TEI in `08_tei_header_with_relaxng/{standoff,flatten,combined}`. The TEI header declares the Soldevila taxonomy; categories are added only for topics actually detected.
14 - Entity/topic CSVs and linkouts live in `data_root/entities/` and `data_root/outputs/`.
15 
21 
22 ### Running the pipeline
23 1) Ensure `data_root/` exists (rename from `data_root_export/` if needed) and contains the per-author folders plus optional PDFs (Index Nominum) and Pleiades CSVs.
24 2) Run scripts in order:  
25    `0` TEI skeletons + verse count (declares Soldevila taxonomy in the header) -> `02/03`  
26    `1` NER into TEI + entity CSVs/index checks → `04`  
27    `2` VIAF, `3` Pleiades, `4` Wikidata (may hit 403) → enriched entities  
28    `5` Topic matching (Soldevila) → CSVs per author (`data_root/topics/`)  
29    `6` Topic annotation (+ Soldevila categories only for detected topics) -> `05` standoff, `06` flatten, `07` combined  
30    `7` XML-model + Relax NG validation → `08_tei_header_with_relaxng/{standoff,flatten,combined}`
52 
53 ### Ejecución
54 1) Asegura `data_root/` (renombrado desde `data_root_export/` si hace falta) con las carpetas por autor y, opcionalmente, PDFs de índices y CSV de Pleiades.
55 2) Corre los scripts en orden:  
56    `0` TEI base + conteo de versos (declara la taxonom?a Soldevila en el header) -> `02/03`  
57    `1` NER + CSV/índices → `04`  
58    `2` VIAF, `3` Pleiades, `4` Wikidata (puede dar 403) → entidades enriquecidas  
59    `5` Tópicos (Soldevila) → CSV por autor (`data_root/topics/`)  
60    `6` Tópicos en TEI → `05` standoff, `06` flatten, `07` combinado  
61    `7` Cabecera xml-model + validación Relax NG → `08_tei_header_with_relaxng/{standoff,flatten,combined}`
62 3) Revisa los TEI finales en `data_root/tei_pipeline/08_tei_header_with_relaxng/`.
84 ### Como executar
85 1) Garanta `data_root/` (renomeado de `data_root_export/` se preciso) com as pastas por autor e, opcionalmente, PDFs de índices e CSV de Pleiades.
86 2) Rode em ordem:  
87    `0` TEI base + contagem de versos (declara a taxonomia Soldevila no header) -> `02/03`  
88    `1` NER + CSV/índices → `04`  
89    `2` VIAF, `3` Pleiades, `4` Wikidata (pode dar 403) → entidades enriquecidas  
90    `5` Tópicos (Soldevila) → CSV por autor (`data_root/topics/`)  
91    `6` Tópicos em TEI → `05` standoff, `06` flatten, `07` combinado  
92    `7` xml-model + validação Relax NG → `08_tei_header_with_relaxng/{standoff,flatten,combined}`
93 3) TEI finais em `data_root/tei_pipeline/08_tei_header_with_relaxng/`.
94 
95 ### Dicionário de Soldevila
96 - Referência: Moreno Soldevila, R. (2011). *Diccionario de motivos amatorios en la literatura latina*. Editorial Universidad de Huelva. http://hdl.handle.net/10272/14398