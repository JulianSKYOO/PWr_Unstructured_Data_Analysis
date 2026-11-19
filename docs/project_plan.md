# Bilingual EU AI Act Document Assistant (RAG, PL/EN)

Group project for the course **Analysis of Unstructured Data** (Wrocław University of Science and Technology).
Instructor: Janusz Szwabiński. The project is 40% of the final grade.

Repository: `github.com/JulianSKYOO/PWr_Unstructured_Data_Analysis`

> **Status:** project setup. The folder structure, configs and notebook skeletons exist; the code is not written yet.
> Everything under "Planned" or marked *proposal* is a draft for the team to confirm. Items marked **TBD** are open decisions (see section 16).

---

## Table of contents

1. [Summary](#1-summary)
2. [Team and roles](#2-team-and-roles)
3. [Research questions](#3-research-questions)
4. [Data](#4-data)
5. [Approach](#5-approach)
6. [How this meets the course requirements](#6-how-this-meets-the-course-requirements)
7. [Evaluation design](#7-evaluation-design)
8. [Interpretability and error analysis](#8-interpretability-and-error-analysis)
9. [Experiment tracking](#9-experiment-tracking)
10. [Repository structure](#10-repository-structure)
11. [Setup and reproducibility](#11-setup-and-reproducibility)
12. [Git workflow](#12-git-workflow)
13. [Roadmap and timeline](#13-roadmap-and-timeline)
14. [Presentation plan](#14-presentation-plan)
15. [Risks and pitfalls](#15-risks-and-pitfalls)
16. [Open decisions](#16-open-decisions)
17. [Deliverables checklist](#17-deliverables-checklist)
18. [Glossary and links](#18-glossary-and-links)

---

## 1. Summary

We build a question-answering assistant for the **EU Artificial Intelligence Act** (Regulation (EU) 2024/1689) that works in **Polish and English**. It uses **Retrieval-Augmented Generation (RAG)**: the system first retrieves the relevant passages of the regulation, then an LLM answers using only those passages.

We study three things:

1. **Chunking.** Splitting the law into fixed-size token windows versus splitting along its natural boundaries (one chunk per Article).
2. **Cross-lingual retrieval.** Can a Polish question find the right passage in the English text, and the reverse?
3. **Hallucinations.** How much does RAG help compared with asking the same LLM without any retrieval, and what does it cost in tokens?

The experiment grid is: **LLM** (several online and local models) × **mode** (no RAG, RAG with fixed chunks, RAG with Article chunks) × **question language** (PL, EN).

```
data (PL + EN regulation text)
   -> preprocessing and structure-aware parsing
   -> representation (TF-IDF/BM25 and multilingual embeddings)
   -> retrieval + LLM generation
   -> evaluation (score, hallucination, tokens)
   -> interpretation and error analysis
```

## 2. Team and roles

| Member | Main responsibility | Course role (guideline) | Main files |
|---|---|---|---|
| **Marysia** (Maria Nowacka) | Prepare the data: extraction, cleaning, structure parsing, PL/EN alignment, both chunk sets, data-prep rules | Data Engineer | `src/data.py`, `src/chunking.py`, notebook 01 |
| **Julian** (Julian Sungkyun Yoo) | Set up the RAG: embeddings, vector index, retriever, prompt, pluggable LLM interface, GPU setup | Modeler | `src/features.py`, `src/retrieval.py`, `src/rag.py`, `src/llms.py`, notebooks 02-03 |
| **Karol** (Karol Cieślik) | Prepare test questions and challenges: question set, reference answers, trap questions | Analyst / Communicator (data side) | `data/eval/`, `src/questions.py` |
| **Michał** (Michał Wrona) | Test no-RAG vs RAG on many online and local LLMs: scores, token usage, final comparison | Analyst / Communicator (results side) | `src/evaluate.py`, `src/interpret.py`, notebooks 04-05 |
| **TBD** | Team Lead: deadlines, milestones, decisions, task board | Team Lead | - |

The guideline recommends one named Team Lead. In a team of four, one member takes this on top of their main task. Decide this explicitly.

Shared duties: everybody commits to the repository, writes their own part of the final notebook, and presents their own slides.

## 3. Research questions

| ID | Question | How we answer it |
|---|---|---|
| RQ1 | Does RAG give more correct and less hallucinated answers than the same LLM without retrieval? | No-RAG vs RAG on identical questions |
| RQ2 | Does Article-level chunking beat fixed-size token chunking? | Same retriever, same LLM, only the chunk set changes |
| RQ3 | How well does cross-lingual retrieval work (PL question, EN document and the reverse) compared with same-language retrieval? | Retrieval metrics per language pair |
| RQ4 | Does a classical lexical retriever (BM25/TF-IDF) match dense multilingual embeddings on legal text? | Same chunks, two retrievers |
| RQ5 | How do models differ in cost (tokens, and money for online models) per correct answer? | Token and latency logging |
| RQ6 | Can we detect hallucinations reliably, for example by checking that the cited Article supports the answer? | Citation check, trap questions, manual audit of a sample |

Working hypotheses (to be tested, not assumed): RAG helps most on exact-detail questions (numbers, deadlines, article references); Article chunks help questions that need a whole provision; fixed chunks cut provisions in the middle and hurt multi-part answers; cross-lingual retrieval is somewhat worse than same-language retrieval but better than lexical matching across languages.

## 4. Data

### 4.1 Source

| Item | Value |
|---|---|
| Document | Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence (the "AI Act") |
| Publisher | EUR-Lex, Official Journal of the EU, L series, 12 July 2024 |
| Identifier | ELI `http://data.europa.eu/eli/reg/2024/1689/oj` |
| Languages | English (EN) and Polish (PL), official translations of the same legal text |
| Files used | `OJ_L_202401689_EN_TXT.pdf` (144 pages) and `OJ_L_202401689_PL_TXT.pdf` |
| Content | Recitals, 113 Articles in 13 Chapters, 13 Annexes |
| Labels | None. The text is unlabelled, so evaluation uses our own question set with reference answers (section 7) |
| Licence | EU legislation published in EUR-Lex may be reused under Commission Decision 2011/833/EU. Licence check was ticked off at Milestone 1; re-confirm the wording before the final submission |

### 4.2 How to obtain the data

Raw files are **not stored in Git**.

1. Open the ELI link above on EUR-Lex.
2. Download the PDF of the Official Journal text in **English** and in **Polish**.
3. Save them as `data/raw/OJ_L_202401689_EN_TXT.pdf` and `data/raw/OJ_L_202401689_PL_TXT.pdf`.

Do not rename the files, so the code can find them.

### 4.3 Data layout

| Folder | Content | In Git? |
|---|---|---|
| `data/raw/` | Original PDFs | No |
| `data/interim/` | Cleaned text per language, parsed structure (recital / article / annex), PL-EN alignment table | No (regenerate with `src/data.py`) |
| `data/processed/` | Chunk files (`chunks_fixed.jsonl`, `chunks_article.jsonl`), BM25/TF-IDF matrices, embeddings, vector indexes | No (regenerate) |
| `data/eval/` | `questions.jsonl` with reference answers | **Yes** (small, part of the experiment) |

### 4.4 Data preparation rules (owner: Marysia, to be refined)

The goal is a clean, reproducible corpus. Every rule below must be written down in `notebooks/01_data_exploration.ipynb`, with the reason for it.

1. **Extract** text with a PDF library and compare against the PDF visually on random pages. Official Journal PDFs use two columns of text in some parts and tables in annexes; check reading order.
2. **Clean** page headers and footers (`Official Journal of the European Union`, `EN`/`PL`, date lines, page numbers), hyphenation at line ends, ligatures, and stray whitespace. Keep a log of what was removed.
3. **Parse the structure** with patterns for `Article N`, `CHAPTER`, `SECTION`, numbered paragraphs, `ANNEX`, and the recitals `(1) ... (180)`. Polish equivalents: `Artykuł N`, `ROZDZIAŁ`, `ZAŁĄCZNIK`. Check counts: 113 Articles, 13 Annexes, 180 recitals in both languages.
4. **Align PL and EN** by Article number and paragraph number, not by sentence similarity. Save an alignment table so cross-lingual evaluation has ground truth.
5. **Attach metadata** to every chunk: `chunk_id`, `lang`, `doc_part` (recital / article / annex), `chapter`, `article`, `paragraph`, `n_tokens`, `source_page`.
6. **Chunking, two variants** (same text, same metadata):
   * **Fixed-size:** windows of N tokens with overlap. N and overlap are chosen on a development question set, not on the test set.
   * **Article-level:** one chunk per Article. Articles longer than the embedding model's limit are split by paragraph and keep the parent Article id.
7. **Token counting** uses one fixed tokenizer for all statistics. State which tokenizer in the notes, because different LLMs count tokens differently.
8. **Do not translate.** The two language versions are official texts; machine translation would add noise.
9. **Record** the final corpus statistics (chunks per variant, token length distribution, empty chunks, duplicates) and save them to `reports/results/`.

## 5. Approach

### 5.1 Pipeline

```
PDF (PL, EN) -> clean text -> parsed structure -> chunk set A (fixed) / chunk set B (Article)
                                                        |
        question (PL or EN) ----------------------> retriever (BM25 | dense)
                                                        | top-k chunks (with ids)
                                                        v
                                   LLM prompt: "answer only from the context, cite the Article,
                                   say 'not in the text' if the answer is missing"
                                                        |
                                                        v
                      answer + cited Articles + token counts + latency -> evaluation
```

### 5.2 Components

| Component | Options (proposal) | Notes |
|---|---|---|
| Lexical retriever | BM25 (`rank-bm25`), TF-IDF (scikit-learn) | Classical, feature-based baseline. Weak across languages by design |
| Dense retriever | Multilingual sentence embeddings such as multilingual-e5, BGE-M3 or paraphrase-multilingual models, with FAISS | Candidates only; **TBD** after a small pilot |
| Retrieval settings | top-k in {3, 5, 10}; optional hybrid (BM25 + dense) | k is tuned on dev questions |
| LLMs | Several **online** models (API) and several **local** models (GPU, CUDA) | **TBD** list. Use the same prompt for all |
| No-RAG baseline | The same question sent to the LLM with no context | Note: the AI Act is public and may be partly in the models' training data, so "no RAG" is not "no knowledge" |
| Prompt | One template per language mode, fixed across models | Versioned in `configs/` |
| Decoding | Temperature 0 (or lowest allowed), fixed max tokens | Reduces noise; log settings for every run |

### 5.3 Experiment grid

| Factor | Levels |
|---|---|
| LLM | N_online online + N_local local (**TBD**) |
| Mode | no RAG; RAG + fixed chunks; RAG + Article chunks |
| Retriever (RAG modes) | BM25; dense |
| Question language | PL; EN |
| Document language | PL; EN (this gives same-language and cross-lingual pairs) |

The full grid is large. Plan: run the full grid on a small pilot, then freeze the best retriever settings on the development questions and run the complete comparison once on the test questions.

## 6. How this meets the course requirements

| Requirement | How we meet it |
|---|---|
| Unstructured data | Raw legal text in two languages (PDF) |
| At least three approaches from different course blocks, including a classical baseline and a neural model | (1) No-RAG LLM baseline; (2) classical retrieval, BM25/TF-IDF + LLM; (3) neural retrieval, multilingual embeddings + LLM. Chunking strategy is an additional factor |
| Interpretability and error analysis | Embedding projections, inspection of failed questions and retrieved chunks, hallucination audit (section 8) |
| Reproducibility | Fixed seeds, fixed question split, one `requirements.txt`, configs in YAML |
| Experiment tracking | MLflow (section 9) |
| Jupyter notebook | Five notebooks in `notebooks/`, final report notebook runs top to bottom |
| GitHub repository | Feature branches + pull requests; no data or checkpoints in Git |
| Presentation | 15 minutes, PDF slides, live demo (section 14) |

## 7. Evaluation design

### 7.1 Question set (owner: Karol)

Stored in `data/eval/questions.jsonl`, one JSON object per line. Every question exists in **both languages** with the same `qid`.

```json
{
  "qid": "Q012",
  "lang": "en",
  "question": "...",
  "type": "numeric",
  "reference_answer": "...",
  "gold_articles": ["99"],
  "answerable": true,
  "split": "test",
  "notes": ""
}
```

Planned question types:

| Type | Purpose |
|---|---|
| `fact` | Single fact stated in one provision |
| `definition` | Terms defined in Article 3 |
| `numeric` | Fines, thresholds, deadlines, dates |
| `multi_article` | Needs two or more Articles |
| `scope` | Who the rules apply to, exemptions |
| `annex` | Answer sits in an Annex (lists, tables) |
| `cross_lingual` | Question in one language, evidence only checked in the other |
| `trap` | **Not answerable** from the text, or based on a false premise. The correct behaviour is to say so. Used to measure hallucination |
| `paraphrase` | Same question phrased in everyday language instead of legal wording |

Rules for the question set:

* Write each question from the text, then have a **second person verify** the reference answer and the gold Article.
* Mix easy and hard questions; record difficulty.
* **Split into `dev` and `test` before running anything.** Use `dev` for choosing chunk size, k and prompt. Use `test` once for the final numbers. Because questions about the same Article are related, split by Article groups (group-aware split) so the same Article does not sit in both sets.
* Target size: **TBD** (enough per type to compare; a few hundred model runs are cheap, thousands of online calls are not).

### 7.2 Metrics

| Level | Metric | Meaning |
|---|---|---|
| Retrieval | Hit@k, Recall@k, MRR against `gold_articles` | Did the right Article come back, and how high? |
| Answer | Correctness score (for example 0 / 0.5 / 1 against the reference answer) | Is the answer right? |
| Faithfulness | Share of answer claims supported by the retrieved context | Detects hallucination even when the answer sounds right |
| Citation | Cited Article is a gold Article | Checks grounding |
| Trap questions | Refusal rate ("not in the text") | Direct hallucination test |
| Cost | Prompt tokens, completion tokens, total tokens, latency; price for online models | Answers RQ5 |
| Language gap | Same metrics split by PL and EN | Answers RQ3 |

Report per-type results and confidence intervals (bootstrap over questions), not only one overall number. Accuracy alone is not enough; show the breakdown.

### 7.3 Scoring method (**TBD**)

Options: (a) manual grading by two people on the full test set; (b) an LLM judge with a fixed rubric; (c) LLM judge plus manual grading of a random sample to measure agreement. Recommended: (c). If an LLM judge is used, it must not be one of the models being compared, and the rubric and judge prompt are stored in the repository.

### 7.4 Evaluation pitfalls specific to this project

* **Training-data contamination.** The AI Act is public. A no-RAG model may know it. Trap questions and exact-number questions expose this.
* **Prompt differences.** Every model gets the same prompt and the same retrieved chunks.
* **Tokenizer differences.** Token usage is read from each provider's usage report, not estimated with one tokenizer, and the context size is compared in characters or chunks as well.
* **Non-determinism.** Fix the seed where the API allows, use temperature 0, and repeat a subset to estimate run-to-run noise.
* **Test-set tuning.** Only the `dev` split may be used to tune anything.

## 8. Interpretability and error analysis

Required by the course, so planned from the start (owners: Michał, Karol).

| Analysis | What we do |
|---|---|
| **Global view** | UMAP or t-SNE of chunk embeddings coloured by Chapter and by language, to show whether PL and EN versions of the same Article land close together. Most informative BM25/TF-IDF terms per Chapter |
| **Local explanations** | For selected questions (correct and incorrect): the retrieved chunks with scores, which context sentences the answer relies on, and word-level attribution for the retriever or a small model where feasible (LIME/SHAP). Attention weights are not used as an explanation on their own |
| **Error analysis** | Inspect all failed test questions. Classify each failure: retrieval miss, chunk cut the answer in half, wrong language match, LLM ignored context, LLM added unsupported facts, ambiguous reference answer |
| **Shortcut check** | Does a model answer correctly without context only because it memorised the text? Does retrieval depend on Article numbers appearing in the question? |
| **Hallucination audit** | Manual check of a random sample of answers for claims not found in the retrieved chunks |
| **Bias and robustness** (bonus) | Results by language, by question type and by difficulty |

Every figure gets 2-3 sentences of interpretation in the notebook.

## 9. Experiment tracking

MLflow is used for every run, including failed ones.

* Experiments: `data-prep`, `retrieval`, `rag-generation`, `final-comparison`.
* Parameters to log: chunk strategy, chunk size, overlap, embedding model, retriever, k, LLM name and provider, prompt version, temperature, seed, question split, question language, document language.
* Metrics to log: Hit@k, MRR, correctness, faithfulness, trap refusal rate, prompt/completion/total tokens, latency, cost.
* Artifacts to log: config file, per-question results table, figures.
* Start the UI with `mlflow ui` and open `http://localhost:5000`.
* `mlruns/` is ignored by Git. Share results by exporting tables to `reports/results/` (small CSV files may be committed).

## 10. Repository structure

```
.
├── data/
│   ├── raw/              # original PDFs                       - not in Git
│   ├── interim/          # cleaned text, parsed structure      - not in Git
│   ├── processed/        # chunks, indexes, embeddings         - not in Git
│   └── eval/             # questions.jsonl                     - in Git
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_features_and_baseline.ipynb
│   ├── 03_deep_models.ipynb
│   ├── 04_evaluation_and_interpretability.ipynb
│   └── 05_final_report.ipynb
├── src/
│   ├── data.py           # extraction, cleaning, structure parsing, PL-EN alignment
│   ├── chunking.py       # fixed-size and Article-level chunking
│   ├── features.py       # TF-IDF / BM25
│   ├── retrieval.py      # dense embeddings, FAISS, cross-lingual retrieval
│   ├── llms.py           # one interface for online and local LLMs
│   ├── rag.py            # prompt + retrieve + generate
│   ├── questions.py      # load and validate the question set
│   ├── evaluate.py       # run the grid, score, log to MLflow
│   ├── interpret.py      # projections, failure inspection
│   └── utils.py          # seeds, device, config loading
├── configs/              # YAML: chunking, retrieval, llms, evaluation
├── models/               # local checkpoints                   - not in Git
├── mlruns/               # MLflow tracking (auto-generated)    - not in Git
├── reports/
│   ├── figures/
│   ├── results/          # exported result tables
│   ├── presentations/    # milestone slides and the project roadmap
│   └── presentation.pdf  # final slides (to be added)
├── docs/                 # course guideline and this project plan
├── requirements.txt
├── .env.example          # variable names for API keys (copy to .env)
├── .gitignore
└── README.md
```

The structure follows the course guideline, with extra files for the RAG components and a versioned `data/eval/` folder.

## 11. Setup and reproducibility

### 11.1 Environment

```bash
git clone git@github.com:JulianSKYOO/PWr_Unstructured_Data_Analysis.git
cd PWr_Unstructured_Data_Analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your own keys locally; never commit .env
```

* **Python version:** **TBD** (agree one minor version for the whole team).
* **GPU:** CUDA on a local machine is arranged. Agree one PyTorch + CUDA build and write it here once decided: `torch == TBD`, `CUDA == TBD`. Members on Colab or Kaggle install the same package versions.
* **API keys:** stored only in `.env` (ignored by Git). The file `.env.example` lists variable names, not values.

### 11.2 Reproducibility rules

* One seed (`42`) set for Python, NumPy and PyTorch through a helper in `src/utils.py`.
* Question split fixed once and saved in `data/eval/questions.jsonl` (the `split` field).
* All settings in `configs/*.yaml`, logged to MLflow with each run.
* Final notebook runs from top to bottom after "Restart kernel and run all". Expensive steps load saved results and the notebook says how they were produced.
* Online LLM answers are cached to disk (in `data/processed/`) so the final report can be rebuilt without new API calls.

### 11.3 Planned commands (not implemented yet)

```bash
python -m src.data          # PDFs -> data/interim
python -m src.chunking      # -> data/processed/chunks_*.jsonl
python -m src.retrieval     # build indexes
python -m src.evaluate --config configs/evaluation.yaml
mlflow ui
```

## 12. Git workflow

```
main        <- stable, presentation-ready code only
  └ develop <- integration branch
      ├ feature/data-prep
      ├ feature/rag-pipeline
      ├ feature/question-set
      └ feature/evaluation
```

* Never commit directly to `main`.
* One feature branch per person and task; merge into `develop` with a Pull Request and at least one review.
* Meaningful commit messages, for example "Add Article-level chunking with paragraph split", not "update".
* **Never commit** raw data, vector indexes, model checkpoints, `.env` files or API keys. If a key is committed by mistake, revoke it immediately; removing it from Git history is not enough.
* Commit at least daily when working; everyone must have commits by Milestone 2.

## 13. Roadmap and timeline

Short version of the project roadmap (also in `reports/presentations/Project_Roadmap.pdf`):

| Step | Owner | Output |
|---|---|---|
| 1. Prepare the data | Marysia | Clean corpus, two chunk sets, data-prep note |
| 2. Set up the RAG | Julian | Working pipeline, swappable LLM interface |
| 3. Prepare test questions and challenges | Karol | `questions.jsonl` with reference answers and trap questions |
| 4. Test no-RAG vs RAG on online and local LLMs | Michał | Results table (score, hallucination, tokens), conclusions |

Steps 1-3 run in parallel after two agreements: the **chunk file format** and the **question file format**. Step 4 needs all three.

### 13.1 Course timeline (from the guideline) and our plan

| Week | Course milestone | Our plan |
|---|---|---|
| 4 | **Milestone 1:** team, repo, topic, data source | Done as listed in the presentation (see 13.2) |
| 6 | Data explored, split fixed, classical baseline working | Clean corpus and both chunk sets; question draft; BM25/TF-IDF + LLM baseline end to end; MLflow active |
| 8 | Tracking active, first neural model | Dense retriever and first local and online LLM runs; question set reviewed |
| 10 | **Milestone 2:** advanced models compared with the baseline | Full pilot grid; pick retriever settings on `dev`; interpretability started |
| 12 | Interpretability and error analysis in progress | Final test run; failure classification; hallucination audit |
| 14 | Results complete, notebook polished, slides drafted | Final report notebook, slides, demo rehearsed |
| 15 | **Presentation** (materials due 48 h before) | 15 minutes with live demo |

Concrete dates for each step: **TBD**.

### 13.2 Milestone 1 status (from the first presentation)

- [x] Team formed and roles assigned
- [x] Communication channel set up
- [x] GitHub repository created
- [x] Project topic chosen and confirmed with the instructor
- [x] Data source identified and licence checked
- [x] `requirements.txt` committed and tested by all members (a draft is now in the repository; re-test after the team edits it)
- [x] GPU access arranged (CUDA, local machine)

### 13.3 Task tracking

Use GitHub Issues (or a shared board). Each task has a description, one owner, a due date and a status (To Do / In Progress / Done).

## 14. Presentation plan

15 minutes including the live demo. Slides are exported as PDF to `reports/presentation.pdf`. Everyone speaks.

| Slide | Content | Time | Speaker (**TBD**) |
|---|---|---|---|
| 1 | Title, team, project name | 0:30 | |
| 2 | Problem and motivation: why a bilingual AI Act assistant, why hallucinations matter in law | 1:00 | |
| 3 | Data: real examples (PL and EN of the same Article), EDA highlights | 2:00 | |
| 4 | Methods: no RAG, BM25/TF-IDF RAG, dense RAG; the two chunking strategies | 2:00 | |
| 5 | Results and model comparison (score, hallucination, tokens) | 2:00 | |
| 6-7 | Interpretability and error analysis, with examples | 3:00 | |
| 8 | **Live demo:** ask a Polish and an English question, plus a trap question; show retrieved Articles | 3:00 | |
| 9 | Conclusions, limitations, lessons | 1:00 | |
| 10 | Q&A | rest | |

Demo rules: never train or index during the demo; load saved indexes. Keep pre-computed answers as a fallback in case of network or GPU problems. Online-model demos need a local fallback in case the API is down.

Questions to prepare for: Is the gain from RAG worth the extra tokens? How does the system behave on another law or another language pair? Can the answer be trusted without checking the cited Article? What would we do with more data or compute?

## 15. Risks and pitfalls

| Risk | Why it matters | Mitigation |
|---|---|---|
| PDF extraction errors (columns, tables, annexes) | Bad text gives bad retrieval for every method | Spot-check pages visually; count Articles, Annexes, recitals |
| PL and EN structure not matching | Breaks cross-lingual ground truth | Align by Article and paragraph number; check counts |
| Question set too small or written by one person | Noisy or biased scores | Second reviewer; per-type counts; bootstrap intervals |
| Tuning on the test questions | Inflated results | `dev` / `test` split fixed before experiments |
| Related questions in both splits | Leakage through the same Article | Group-aware split by Article |
| LLM judge favours its own family or style | Biased scoring | Judge not among compared models; manual sample check |
| API costs and rate limits | Runs stop or budget is exceeded | Caching, pilot first, token budget set up front |
| Models differ in context size and tokenization | Unfair comparison | Same prompt and chunks; report provider-reported tokens |
| Compute planned too late | No time to run local models | Estimate runtime on a small pilot in week 6 |
| One person does everything | Others learn nothing; burnout | Issues with owners, reviews on pull requests |
| Notebook does not run from scratch | Fails at submission | Restart and run all before every merge to `main` |
| Secrets in Git | Leaked API keys | `.env` ignored; review diffs before pushing |

Technical pitfalls from the course guideline that apply here: fit any TF-IDF vocabulary on the training side only; use each model's own tokenizer when running it locally; set seeds; start with a baseline; look at raw data and raw model outputs, not only at scores; debug on a few questions before the full run.

## 16. Open decisions

| # | Decision | Recommendation | Owner |
|---|---|---|---|
| 1 | Team Lead | One person, named now | Team |
| 2 | List of LLMs (how many online, how many local, which) | 3-4 online and 3-4 local, covering small to large sizes | Michał + Julian |
| 3 | Embedding model(s) | Pilot two multilingual models on `dev`, keep one | Julian |
| 4 | Fixed chunk size and overlap | Choose on `dev`; try two sizes | Marysia + Julian |
| 5 | Number of questions and split ratio | Enough for several questions per type; split by Article | Karol |
| 6 | Scoring method | Manual sample + LLM judge outside the compared set | Michał + Karol |
| 7 | Python, PyTorch and CUDA versions | Pin them in `requirements.txt` and this README | Julian |
| 8 | Keep source PDFs out of Git, or commit them (public, about a few MB each)? | Keep out; document the download | Team |
| 9 | Step deadlines | One date per step, set at the next meeting | Team |
| 10 | Task board (GitHub Issues or other) | GitHub Issues | Team Lead |

## 17. Deliverables checklist

Final week (all materials due **48 hours before the presentation**):

- [ ] Final notebook runs cleanly from top to bottom (loading saved results where training or API calls are expensive)
- [ ] All results logged in MLflow
- [ ] Presentation exported as PDF to `reports/presentation.pdf`
- [ ] Live demo rehearsed at least once, with a fallback
- [ ] Repository clean: no data, checkpoints or keys; README up to date; every member has commits
- [ ] Everything submitted on time

Intermediate checks (from the guideline):

- [ ] Week 6: data in `data/raw/`, README explains how to obtain it, EDA notebook complete, split fixed, baseline end to end, MLflow active
- [ ] Week 10: at least two advanced approaches compared with the baseline, all experiments logged, interpretability started, everyone has commits
- [ ] Week 14: final comparison table, written interpretability section, polished notebook, drafted slides

## 18. Glossary and links

| Term | Meaning |
|---|---|
| RAG | Retrieval-Augmented Generation: retrieve relevant text first, then let an LLM answer from it |
| Chunk | A piece of the document that is indexed and retrieved |
| Fixed-size chunking | Cutting the text into windows of N tokens, ignoring document structure |
| Article-level chunking | Cutting along the law's own structure, one chunk per Article |
| Cross-lingual retrieval | Finding text in one language with a question in another |
| BM25 / TF-IDF | Classical word-based ranking methods |
| Dense retrieval | Ranking by similarity of neural embeddings |
| Hallucination | An answer containing claims not supported by the source |
| Trap question | A question whose correct answer is "this is not in the text" |
| Hit@k / MRR | Retrieval metrics: right chunk in top k / average inverse rank of the right chunk |
| Token | Unit of text an LLM reads and bills by |
| ELI | European Legislation Identifier |

Links:

* Course guideline: `docs/group_project_guidelines.html`
* Regulation (EU) 2024/1689 on EUR-Lex: `http://data.europa.eu/eli/reg/2024/1689/oj`
* First presentation and roadmap: `reports/presentations/`
