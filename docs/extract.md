# Project Extract — Movie Recommender

> **Reading rule:** Every claim below is tagged **[DONE]** (verified in current code) or **[PLANNED]** (only in `completion-plan.md`, not built yet). Write CV bullets from **[DONE]** only.

---

## 1. Identity

- **Project name:** Movie Recommender System
- **One-line description:** Content-based movie recommender that suggests 5 similar films from a chosen title using NLP tag similarity.
- **Type:** Personal project (self-initiated; no git history, no PS/coursework markers found).
- **Current completion %:** ~40% toward CV-ready/deployed. *ML core works locally; everything around it (deploy, repo, README, posters) is missing.*
- **Target completion per plan:** 100% = runs end-to-end with posters, deployed at a public Streamlit URL, on GitHub, with README + training notebook. [PLANNED]

---

## 2. The Problem

- **What it solves:** Helps a user discover what to watch next. Given one movie they like, it returns 5 similar movies based on content (genres, keywords, cast, director, plot) rather than requiring ratings data or user history. It's a cold-start-friendly, content-based approach over ~4,800 films.
- **Target user/audience:** Casual movie watchers deciding what to watch next; also a portfolio demo of an NLP/recommender pipeline for recruiters.

---

## 3. Technical Stack

**[DONE] — currently wired up in code:**
- Python
- `streamlit` — UI (`st.title`, `st.selectbox`, `st.button`, `st.write`) — `app.py`
- `pandas` — DataFrame from the movies dict — `app.py`
- `pickle` — loads `movies_dict.pkl` + `similarity.pkl` — `app.py`
- Precomputed **cosine-similarity matrix** (4806×4806) shipped as `similarity.pkl`
- Dataset: 4,806 movies with `['movie_id_x', 'title', 'tags']`, tags pre-stemmed (Porter)
- ⚠️ **No `requirements.txt`** exists — versions are NOT pinned anywhere. Stack above is inferred from imports.

**[PLANNED] — to be added per completion-plan.md:**
- `scikit-learn` — `CountVectorizer` + `cosine_similarity` recomputed at runtime (replaces the 184 MB pickle)
- `requests` — TMDB API calls for poster images
- TMDB API (poster fetching via `movie_id_x`), key stored as a Streamlit secret
- `requirements.txt` pinning the above
- Jupyter `notebook.ipynb` reproducing the preprocessing + model

---

## 4. Architecture & Key Decisions

**[DONE] How it's currently built:**
- Single file `app.py` (29 lines). Flow: `pickle.load` both files → build `pd.DataFrame` → `st.selectbox` of titles → on button click, `recommend()` looks up the movie's row index, reads its similarity vector, sorts descending, takes indices `[1:6]`, returns those titles → printed as plain text via `st.write`.
- Model artifact (`similarity.pkl`, 184 MB) is precomputed offline and loaded at runtime — i.e. the data-science work is frozen in a binary; the preprocessing/training code is **NOT in the repo**.
- Pattern: classic content-based filtering; no API layer, no DB, no tests, no caching.

**[PLANNED] Architecture changes per plan:**
- Delete `similarity.pkl`; recompute similarity at startup from `tags` via scikit-learn, cached with `@st.cache_data`/`@st.cache_resource`.
- Add poster grid (`st.columns(5)` + TMDB images), error handling, and a reproducibility notebook.
- Move secret handling to `st.secrets`.

**Scale indicators that exist NOW:**
- `app.py`: **29 lines** (`wc -l`).
- Source files: **1** (`app.py`). Data files: 2 pickles (`movies_dict.pkl` 2.1 MB, `similarity.pkl` 184 MB).
- **Git commits: 0** — no git repo initialized.
- Dataset: **4,806 movies**, **3 columns**, similarity matrix **4806×4806**.

---

## 5. Results / Metrics

**[DONE] Real numbers that exist today:**
- Dataset size: **4,806 movies** (verified).
- Similarity matrix: **4806×4806** (verified).
- Returns **top-5** recommendations per query (verified in `recommend()`).
- ❌ No accuracy/latency/quality metric is measured anywhere. This is an unsupervised content-based model — there's no labeled ground truth, so "accuracy" isn't applicable, and **latency is not instrumented**.

**[PLANNED] Metrics claimable once plan is complete — and what to instrument:**
- **Recommendation latency (ms):** wrap `recommend()` (and poster fetch) in `time.perf_counter()`; display "recommended in X ms" (Nice-to-Have #4). *This is the one number worth capturing.*
- **Cold-start / matrix build time:** time the runtime `cosine_similarity` computation once recompute-at-startup lands (MUST-DO #1).
- **Corpus/vocab size:** report vocabulary size from `CountVectorizer(max_features=5000)` once the notebook exists (MUST-DO #5).

**If nothing is measured yet:** Correct — nothing is. Start capturing **latency (ms)** and **matrix build time** as soon as you implement MUST-DO #1 and Nice-to-Have #4.

---

## 6. CV Material — STRICT

### Resume bullets from [DONE] work only

- **Data Scientist:** "Built a content-based movie recommendation engine over **4,806 films**, modeling each title as an NLP tag vector and ranking the **top-5** most similar movies via cosine similarity (Python, pandas, Streamlit)." *(Verifiable today. Caveat: the preprocessing/training code isn't in the repo yet, so be ready to explain the pipeline verbally; the notebook (MUST-DO #5) makes this fully defensible.)*
- **Data Analyst:** Borderline. Usable: "Built an interactive Streamlit app surfacing similar-movie recommendations across a **4,806-record** film dataset." Thin — it shows tool familiarity, not analysis/insight. Prefer to strengthen via TMDB metadata (Nice-to-Have #2) before leaning on this.
- **SDE:** **Not enough built yet for an SDE bullet** — 29 lines, one function, no API/tests/architecture/deployment. Needs deployment (MUST-DO #1–7) at minimum before any SDE-flavored claim.

### Future bullets (DO NOT use yet) — ranked by CV payoff

- **Unlocks the strongest bullet — MUST-DO #1–7 (ship + deploy):** "Built and **deployed** a content-based movie recommender (Streamlit, scikit-learn) over 4,806 films with a poster-grid UI; live at [URL]." → unlocked by deployment + posters.
- **MUST-DO #5 (training notebook):** "Engineered NLP features (genres, cast, keywords, overview) into a CountVectorizer + cosine-similarity model, documented end-to-end in a reproducible notebook." → makes the DS bullet fully defensible.
- **Nice-to-Have #4 (latency):** adds "...returning recommendations in **[X] ms**" to the DS/SDE bullet.
- **Nice-to-Have #3 (TF-IDF compare):** "Compared CountVectorizer vs TF-IDF vectorization and evaluated effect on recommendation relevance." → a real DS talking point.

---

## 7. For the Portfolio Website

- **Card title:** "Movie Recommender — NLP Content-Based Engine"
- **2-sentence summary:** "A content-based recommender that suggests 5 similar films from any chosen title using NLP tag vectors and cosine similarity over ~4,800 movies. Built with Python, scikit-learn, and Streamlit, with a poster-grid UI." → ⚠️ **The live version will initially show less than this card implies** until MUST-DO ships: today it returns plain-text titles, no posters, and is not deployed.
- **Live demo:** [PLANNED] Streamlit Community Cloud (free tier, ~1 GB RAM — fits the matrix). **Fully deployable on a free tier; no GPU/LLM required.** Not deployable yet (needs MUST-DO #1, #4, #7 first). No live URL exists today.
- **Screenshot/visual to capture (once built):** the poster grid of 5 recommendations for a well-known movie (e.g. "Avatar" → results), plus a short GIF of select-and-recommend.
- **README status:** **NOT FOUND** — no README exists. Needs to be written (MUST-DO #6) before linking anywhere.

---

## 8. Build Priority Signal

- **High-CV-value vs polish:** High-value = deployment + GitHub + notebook + posters (MUST-DO #1–7) — these convert a local script into a clickable, defensible portfolio piece. Polish = explainability, TF-IDF compare, metadata cards (Nice-to-Haves). Roughly **70% of remaining planned work is high-CV-value**, 30% polish.
- **Honest take:** Worth finishing, but as a **quick win, not a flagship**. It's a very common beginner DS project, so it won't differentiate you alone — its value is being *shipped and clickable*, which most candidates lack. Likely **lower priority than a more original DS/DA project**, but the effort-to-payoff here is excellent because the ML core already works.
- **Effort to first CV-usable state:** ~**6–8 focused hours** (MUST-DO + deploy). The first CV-usable milestone is specifically: deployed URL + posters + README + notebook.

---

## 9. Cleanup / Secrets Check

- **Hardcoded secrets:** ✅ **None found.** Grep for api_key/token/secret/password/Bearer/AWS/GitHub/Stripe-style keys and `user:pass@host` URLs in `app.py` returned no matches. No secrets to rotate. *(Keep it that way: when TMDB is added, use `st.secrets`, never a literal.)*
- **Should be cut / hidden:**
  - `similarity.pkl` (**184 MB**) — cut; exceeds GitHub's 100 MB limit and blocks push. Recompute at runtime (MUST-DO #1).
  - `venv/` (9,000+ files) sitting in the project — must be `.gitignore`d before any commit.
  - Personal string `'How can i help You Binu?'` in `app.py` — remove before the repo/app is public.
  - Dead/sloppy lines after the `return` in `recommend()` — clean up.

---

*Source files read: `app.py`, `movies_dict.pkl` (structure), `similarity.pkl` (shape), `docs/completion-plan.md`. No README, requirements.txt, package.json, or git history exist in the repo.*
