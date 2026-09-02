# Figures

Hand-authored diagrams, rendered at 1600 px wide. All six are embedded in the top-level
[README](../../README.md); this index is the reference for what each one claims, and the
place to look when a figure needs updating.

## Architecture and storage

| File | Shows |
|---|---|
| `01-architecture.png` | The whole system end to end — five sources, the ingestion layer, three transformation paths, three stores, four MCP tools, one agent. Store-to-tool wiring is drawn rather than described, so it is visible which tool reaches which backend. |
| `02-etl-vs-elt.png` | The position of the load boundary in each of the three paths. Everything left of the dashed line happens before the data reaches the store; everything right of it happens inside. That position is the entire ETL/ELT distinction. |
| `03-star-schema.png` | The PostgreSQL star schema at review grain — four dimensions around `fact_review`, `fact_review_weather` as a Kimball bridge holding up to ten daily weather rows per review, and `noise_borough_month` as a pre-aggregate joined on natural keys rather than a surrogate. |

## Vector layer

| File | Shows |
|---|---|
| `04-vector-index.png` | How both ChromaDB indexes are built — two-level stratified sampling down to 50,000 reviews, paragraph chunking for Wikipedia intros, and the single encoder shared between index time and query time. |

## Agent and evaluation

| File | Shows |
|---|---|
| `05-agent-paradigms.png` | Single-agent ReAct beside the planner–executor–synthesiser graph. The claim is the size of the action space per model call: one box doing three jobs, versus three nodes doing one each. Measured results sit under each panel. |
| `06-evaluation.png` | Quality metrics on a shared 0–1 axis and latency on its own axis, across the three paradigms. Deliberately two charts rather than one with two scales. |

`00-airbnb-logo.png` is not a figure — it is the wordmark used in the README header.

## Regenerating

The diagrams are hand-authored SVG rendered to PNG at 1600 px. Source SVGs are not kept in
the repository — the PNGs are the artefact, and a figure that needs changing is redrawn
rather than patched.

## Style

One palette across all six, so the stores are the same colour everywhere they appear:

| | Colour | Used for |
|---|---|---|
| PostgreSQL | blue `#2a78d6` | the ETL path, the warehouse, `query_warehouse` |
| MongoDB | aqua `#1baf7a` | the geospatial ELT path, `query_geodata` |
| ChromaDB | orange `#eb6834` | the vector ELT path, both ChromaDB tools |
| Facts / bridge | red `#e0484d` | fact tables in the star schema |

The categorical hues were checked for colour-vision separation before use, and every mark
carries a text label as well as a colour, so no figure depends on colour alone.
