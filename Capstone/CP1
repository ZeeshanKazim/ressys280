# RouteGraph-RAG: Constraint-Aware Two-Tower Flight Recommender with Review-Text Signals

*Tagline:* **The Core of Modern RecSys: RAG + Two-Tower Architecture** for policy-aware corporate flight recommendations.

This project builds on **FlightRank 2025: Aeroclub RecSys Cup (Kaggle)** data to recommend flights for corporate travellers under policy and preference constraints. The core idea is a **two-tower / classic ranker** for candidate retrieval on FlightRank, enriched with (i) **route-graph signals** (popular connections, alliances/mileage groups) and (ii) **RAG** over external review text (e.g., TripAdvisor/airline service reviews) to inject qualitative evidence into ranking. An **LLM is used only as a policy-aware, constraint-aware re-ranker/explainer**—not as a chat UI.

> **Name change:** Earlier working title “FlightAgent-LLM” is refined to **RouteGraph-RAG** after improvement to emphasize model-centric contributions (graphs + RAG + two-tower).

---

## 1) Progress on Milestones

**From proposal:**
- Finish EDA on FlightRank 2025.  
- Implement a baseline ranker (e.g., XGBoost or simple two-tower).  
- Define a prompt format & zero-shot LLM re-rank for a single query.

### Status by milestone

#### A) EDA on FlightRank 2025 — **Status: Completed**
- **Data processed:** Loaded `train.csv` / `test.csv` (interactions with labels & option features), plus auxiliary CSVs: routes, airports, and airline reviews (external). Cleaned obvious missing values; normalized carrier codes (IATA), standardized airport IDs, parsed datetimes; created **train/val/test splits by query** (to avoid leakage).
- **Feature engineering:** Derived **red-eye flag** (depart 22:00–05:00), **time-bucket** (hour-of-day), **day-of-week**, **days-to-departure**, **stops**, **log(price)**, **duration_minutes**, **route key** `(origin,dest)`, **carrier–route key** `(carrier,origin,dest)`.
- **EDA findings:**
  - Heavy-tailed prices → **log transform stabilizes**.
  - Duration correlates with stops; **non-linear effect** suggests tree/ranker rather than pure linear.
  - **Long-tail options per query**; most queries have many candidates but few positives → motivates **efficient candidate retrieval** (two-tower) and **listwise ranking**.
  - **Carrier heterogeneity across routes**; motivates adding **route-graph priors** and **review-text signals** (service/food/punctuality).
- **Design implication:** Use **route popularity / reliability** via a route graph; inject **qualitative priors** (comfort/food/staff) from reviews via **RAG** at re-rank stage.

#### B) Baseline ranker — **Status: Completed (baseline running)**
- **Model:** Gradient-boosted ranker (**XGBoost**, `rank:pairwise`) as a strong tabular baseline before adding two-tower.
- **Features (current):** `log_price`, `duration_minutes`, `stops`, `red_eye`, `depart_hour`, `dow`, `days_to_departure`, `carrier_onehot` (top-N), `origin_onehot` (top-N), `dest_onehot` (top-N).
- **Training progress:** Trained on train; early validation metrics: **[NDCG@10 ≈ 0.XX]**, **[Recall@10 ≈ 0.XX]** *(placeholders—exact numbers to be inserted after full sweep)*. Hyperparams roughly tuned (100–300 trees, depth 6–8, eta 0.05–0.1).
- **Note:** This baseline **does not** use RAG or review text yet; it serves as the **reference** for later RouteGraph + RAG + LLM improvements. A **two-tower** retriever (user/query tower vs. item/option tower) is prototyped on IDs/context features and planned to replace/augment candidate generation at CP2.

#### C) Prompt format & zero-shot LLM re-rank (prototype) — **Status: Partially completed**
- **Prompt template (single query, zero-shot):**
  - **User/Policy block:** JSON with hard rules (max price, no red-eye, prefer alliance, max stops) + soft prefs (earlier arrival, SkyTeam if similar).
  - **Candidates block:** Top-K from baseline (K≈20) summarized as `{id, carrier, dep_time, arr_time, stops, price, duration, route, alliance?}`.
  - **Instruction:** “Re-order these options to best satisfy user policy and preferences. **Do not invent flights.** Output a JSON array with `{id, score, reason}`; **explain top-3** briefly.”
- **Execution:** Tested on toy queries with constrained, deterministic output; **hard filters** (e.g., price cap, red-eye ban) applied **before** LLM; LLM only **re-orders** the valid set.
- **Status note:** Works on single queries; will be extended with **review-snippet RAG** and **route-graph** features as additional context at CP2.

---

## 2) Problem Solving & Adaptation (2 pts)

### Lecturer feedback (interpreted)
- Initial plan looked like **“ChatGPT UI on top of existing flight portals”**; unclear recommender contribution.  
- To be RecSys-worthy, incorporate data **no portal systematically uses**, e.g., **alliances/mileage groups**, **user & flight reviews** (food/service/punctuality), and **route structure**.  
- Focus **less on chat interface**, **more** on models and **new signals**.

### Adaptation to RouteGraph-RAG
- **From:** FlightAgent-LLM (chat-forward, modest reranker).  
- **To:** **RouteGraph-RAG (model-centric)**:
  - **Backbone retrieval/ranking:** baseline **XGBoost** now, **two-tower** retrieval next.
  - **Route graph:** Airports ↔ Routes ↔ Carriers; compute **popularity**, **reliability**, **carrier presence**, and later integrate **alliances/mileage groups**.
  - **RAG over reviews:** Index airline/route reviews; retrieve **top-k snippets** per candidate to add **qualitative signals** (comfort/food/staff/punctuality).
  - **LLM role:** Strictly **constraint-aware re-ranker/explainer** that **fuses** numeric features + route-graph priors + review snippets; **no hallucinated flights**.

### Concrete challenges & solutions
- **Aligning external reviews to FlightRank options**  
  **Solution:** Map by `(carrier, origin, dest)`; if route-level sparse, fall back to **carrier-level aggregates**; store per-key **sentiment summaries** for RAG.
- **Missing alliance/mileage info in base data**  
  **Solution:** Add a tiny **`alliances.json`** mapping `carrier → {SkyTeam/Star/Oneworld/None}`; verified codes against carriers present in data.
- **Avoiding LLM hallucinations**  
  **Solution:** Apply **hard filtering first**; prompt explicitly forbids **new flights/prices**; require output **IDs from provided candidate set**.
- **Keeping CP1 model-focused (not UI)**  
  **Solution:** Deferred UI entirely; invested time in **data cleaning**, **baseline training**, **prompt design**, and **RAG/schema planning**.
- **Choosing a practical baseline**  
  **Solution:** Start with **XGBoost ranker** (fast, interpretable, tabular-friendly); add **two-tower** for scalable retrieval at CP2.

---

## 3) Next Steps

- **Integrate route-graph features** (Personalized PageRank/degree/reliability, alliance preference scores) directly into the ranker features.  
- **Implement full RAG pipeline:** embed review text; for each candidate, retrieve **top-k snippets** to condition the LLM re-ranker (and optionally compute a **lightweight text-match score** for hybrid ranking).  
- **Add two-tower retrieval:** build query/context and item/option embeddings for **fast candidate generation**, feeding top-N to ranker/LLM.  
- **Evaluate vs. baseline** on validation: **NDCG@10, Recall@10**, and **constraint-violation rate** (hard policy checks).  
- *(Later)* **Minimal UI:** a **two-tab browser demo**—**Graph** (RouteGraph-PPR + constraints) and **Text Hybrid** (review-aware MMR)—**only after** modeling results are in.

---
