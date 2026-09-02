# Figures

Every figure here is taken from the project report itself — cut out of the submitted PDF at
its native resolution, not redrawn. All four are embedded in the top-level
[README](../../README.md); this index records where each one came from and what it shows.

| File | Report source | Shows |
|---|---|---|
| `01-transformation-paths.png` | Figure 1, p.11 | The five sources fanning out into three transformation paths — PySpark ETL into the PostgreSQL star schema, Python ELT into the MongoDB geospatial store, Sentence-Transformers encoding into the ChromaDB indexes. |
| `02-star-schema.png` | Figure 2, p.38 (appendix) | The PostgreSQL star schema at review grain — four dimensions around `fact_review`, `fact_review_weather` as a Kimball bridge, and `noise_borough_month` as a pre-aggregate joined on natural keys. |
| `03-react-baseline.png` | Figure 2, p.21 | Paradigm I: one local model in a Reason–Act–Observe loop doing routing, parameter generation and synthesis in a single iteration over the four MCP tools. |
| `04-planner-executor-synthesizer.png` | Figure 3, p.23 | Paradigms II and III: the same tools decomposed into planner, three executors and a synthesiser, with the two compound routes drawn as conditional edges. |

`00-airbnb-logo.png` is not a figure — it is the wordmark used in the README header.

## How they were produced

Each figure was extracted from the report PDF as its embedded image, then colour-inverted
for a dark background:

- lightness is inverted, **hue is preserved** — PostgreSQL stays blue, MongoDB stays green,
  and both logos keep their real colours, which a straight RGB inversion would flip;
- neutral greys are lifted further than saturated colours, so the labels and connectors that
  were mid-grey on white read as near-white rather than staying grey;
- the page background is pinned to true black, so all four figures sit on the same ground.

A figure that needs changing is re-cut from the report and re-processed, not patched.
