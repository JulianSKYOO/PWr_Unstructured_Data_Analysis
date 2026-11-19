# Bilingual EU AI Act Document Assistant (RAG, PL/EN)

Group project for the course Analysis of Unstructured Data (Wrocław University of Science and Technology).
We build a question-answering assistant for the EU AI Act (Regulation (EU) 2024/1689) that works in Polish and English using Retrieval-Augmented Generation (RAG).
We compare fixed-size token chunking with Article-level chunking, test cross-lingual retrieval, and check how far RAG reduces hallucinations.
Each question goes to several online and local LLMs, both with and without retrieval, and we record answer scores and token usage.
Team: Karol Cieślik, Maria (Marysia) Nowacka, Michał Wrona, Julian Sungkyun Yoo. The detailed plan is in [docs/project_plan.md](docs/project_plan.md).
