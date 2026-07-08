/
# Movie Recommender — Completion Plan

**Project type:** Content-based movie recommender (NLP + cosine similarity), Streamlit UI.
**Primary target role:** Data Scientist (DS). Secondary: Data Analyst. *Not* SDE.
**Verdict up front:** Finishable in ~6–8 focused hours to CV-ready + deployed. Worth it as a *modest, shipped* project — not a flagship. Ship it fast, then move on.

---

## 1. Current State (verified from code)

**What works right now:**
- `app.py` (35 lines, Streamlit): loads pickles, `recommend(movie)` finds the selected movie's row, sorts the similarity vector, returns top-5 titles. Runs locally.
- `movies_dict.pkl`: 4,806 movies, columns `['movie_id_x', 'title', 'tags']`. `tags` is already cleaned + Porter-stemmed text (overview + genres + keywords + cast + director). `movie_id_x` = TMDB movie ID (enables poster fetching).
- `similarity.pkl`: precomputed 4806×4806 cosine-similarity matrix. The ML core is done and correct.

**What's stubbed / broken / incomplete:**
- Output is plain text titles only — no posters, no layout. Looks like a script, not a product.
- Hardcoded personal string in the UI: `'How can i help You Binu?'` — must be removed before it's public/on a CV.
- No error handling (e.g., movie not found → crash).
- `recommend()` has dead code (blank `return`-after-return indentation is fine, but the trailing blank lines/structure is sloppy).

**What's completely missing:**
- No `requirements.txt`, no `README.md`, no `.gitignore`, **no git repo at all**.
- `venv/` is sitting inside the project folder (9,000+ files) and would get committed without a `.gitignore`.
- No deployment, no public URL.
- No training notebook / reproducibility artifact showing the DS work (vectorization, similarity) — right now the "data science" is invisible, frozen inside a binary pickle.
- **Blocker:** `similarity.pkl` is **184 MB** → exceeds GitHub's 100 MB file limit and bloats any deploy.

---

## 2. Definition of "CV-Ready Done" for THIS project

The minimum bar. Anything past this is optional:

1. **Runs end-to-end** on the sample data: pick a movie → get 5 recommendations **with poster images** in a clean grid.
2. **Deployed at a public URL** (Streamlit Community Cloud) that a recruiter can click and use in <10 seconds — no setup.
3. **On GitHub**, public repo, with the 184 MB pickle removed (similarity recomputed at load), so the repo actually pushes.
4. **README** with: one-line description, the approach (how it works), a screenshot/GIF of a real recommendation, live demo link, and local-setup steps.
5. **A short training notebook** (`notebook.ipynb`) checked in that reproduces `movies_dict` and the similarity matrix from raw data — this is what makes it a *DS* project and not a black box.
6. No personal/hardcoded strings; TMDB API key handled as a secret, never committed.

---

## 3. MUST-DO (to reach CV-ready)

Ordered. Only what's genuinely required for Section 2.

1. **Kill the 184 MB pickle; recompute similarity at startup.** `S`
   - In `app.py`, on load: take `movies['tags']` → `CountVectorizer(max_features=5000, stop_words='english')` → `cosine_similarity`. Wrap in `@st.cache_data`/`@st.cache_resource` so it computes once. Delete `similarity.pkl` from the repo.
   - Files: `app.py` (+ delete `similarity.pkl`).
   - Why: A 184 MB binary can't go on GitHub and makes deploy fail/slow. Recomputing from `tags` (a few seconds) makes the repo lightweight *and* shows you understand the pipeline, not just a frozen artifact.

2. **Add poster images + grid layout.** `M`
   - Fetch posters from TMDB API using `movie_id_x` (`/movie/{id}` → `poster_path`). Render 5 recommendations in `st.columns(5)` with image + title. Cache responses.
   - Files: `app.py`.
   - Why: This is the single biggest visual upgrade. Text titles look like homework; a poster grid looks like a product an interviewer will actually click through.

3. **Remove personal strings + add error handling.** `S`
   - Replace `'How can i help You Binu?'` with `'Select a movie'`. Guard `recommend()` against missing titles. Clean the dead lines.
   - Files: `app.py`.
   - Why: Anything public/on a CV must look professional; a crash during an interview demo is fatal.

4. **Add `requirements.txt` + `.gitignore`.** `S`
   - `requirements.txt`: `streamlit, pandas, scikit-learn, requests`. `.gitignore`: `venv/`, `*.pkl` exceptions handled (keep `movies_dict.pkl`, ignore `similarity.pkl`), `.streamlit/secrets.toml`.
   - Files: new `requirements.txt`, `.gitignore`.
   - Why: Required for Streamlit Cloud to build, and to avoid committing the 9k-file venv.

5. **Add the training notebook (reproducibility).** `M`
   - `notebook.ipynb`: load TMDB 5000 movies+credits CSVs → merge → extract genres/keywords/cast(top 3)/director/overview → build `tags` → stem → `CountVectorizer` → `cosine_similarity` → export `movies_dict.pkl`. Markdown explaining each step. (You can reverse-engineer this from the existing `tags` format — it's standard TMDB-5000 preprocessing.)
   - Files: new `notebook.ipynb`.
   - Why: This *is* the data-science evidence. Without it, the project is a UI over a mystery binary; with it, you can speak to feature engineering and similarity in an interview.

6. **Write the README.** `S`
   - Description, "How it works" (3 lines), live demo link, screenshot/GIF, local setup, tech stack.
   - Files: new `README.md`.
   - Why: First thing a recruiter reads. Demo link + screenshot do 80% of the selling.

7. **Init git + push to GitHub.** `S`
   - `git init`, commit, push to a new public repo. Confirm repo size is small (no 184 MB file).
   - Why: No GitHub = invisible. Also the deploy source.

---

## 4. Deployment Plan (required)

**Recommended host: Streamlit Community Cloud — free, and purpose-built for exactly this (a `streamlit run app.py` app).** Zero Docker, deploys straight from a GitHub repo. No reason to use Render/HF Spaces here.

**Exact steps:**
1. Finish MUST-DO 1–7 (repo must build without the 184 MB pickle and with `requirements.txt`).
2. Go to share.streamlit.io → "New app" → connect GitHub → pick repo/branch → main file `app.py` → Deploy.
3. First boot recomputes the similarity matrix from `tags` (cached after). Expect ~10–30 s cold start, then instant.

**Secrets / API keys (TMDB):**
- Do **not** hardcode the TMDB key. Read it via `st.secrets["TMDB_API_KEY"]`.
- Locally: put it in `.streamlit/secrets.toml` (which is in `.gitignore`).
- On Streamlit Cloud: paste it in the app's **Settings → Secrets** UI. Never commit it.
- Get a free TMDB API key at themoviedb.org (free, instant).

**Cost:** $0. Streamlit Community Cloud free tier (1 GB RAM — the 4806×4806 float matrix ≈ 185 MB in memory, comfortably fits) and TMDB API are both free. Nothing here costs money.

**Fallback:** None needed — this deploys live for free. (No GPU/LLM dependency.) Still record a 15-second GIF for the README so the project demos even if the app is cold-starting.

---

## 5. NICE-TO-HAVE (only after MUST-DO + deploy are shipped)

Do **not** start these until there's a live URL. Ranked by impact-per-effort:

1. **Show *why* it was recommended** (top shared tags/genres between selected and recommended movie). `S` — Adds explainability, a real DS talking point. Impresses: DS.
2. **Movie metadata on cards** (rating, release year, overview snippet from TMDB). `S` — Makes it feel complete. Impresses: DA/DS.
3. **Swap CountVectorizer → TF-IDF and compare** (note the difference in a notebook cell). `M` — Shows you understand vectorization trade-offs. Impresses: DS.
4. **Latency display** ("recommended in X ms"). `S` — Gives you the metric for your resume bullet. Impresses: DS/SDE.

---

## 6. What to CUT or DESCOPE

- **`similarity.pkl` (184 MB): cut entirely.** Recompute at runtime. It's the single biggest liability — blocks GitHub, bloats deploy, and adds nothing you can't regenerate in seconds.
- **`venv/` in the repo: never commit.** `.gitignore` it.
- **Don't build a "real" recommender** (collaborative filtering, embeddings, a backend API, a database). Tempting, but it turns a 6-hour ship into a multi-day rebuild for a project that should stay modest. Resist scope creep — ship the content-based version.
- **Don't add user accounts / ratings / "movies you liked" history.** Over-engineering for a portfolio demo.

---

## 7. Resume Bullet Preview

> **Built and deployed a content-based movie recommender** (Streamlit, scikit-learn) over **4,806 films**, engineering NLP tag features (genres, cast, keywords, overview) into a CountVectorizer + cosine-similarity model that returns top-5 recommendations with posters in **[X] ms**; live at **[demo URL]**.

→ **Metric to instrument:** recommendation latency in ms (Nice-to-Have #4). Capture it once and drop it in.

---

## 8. Total Effort Estimate

| Bucket | Effort |
|---|---|
| MUST-DO 1–7 | ~5–6 h |
| Deployment | ~1 h |
| **Total to CV-ready + deployed** | **~6–8 focused hours** |

**Worth finishing?** Yes — but as a *quick win*, not a centerpiece. It's a common beginner DS project, so it won't carry your portfolio alone; its value is being *shipped, public, and clickable*, which most candidates' projects are not. 6–8 hours to a live, polished demo + a clean GitHub repo + a real notebook is a strong return. Do it in one focused day, then put your deeper effort into a more differentiated DS/DA project. Do **not** spend a week here.
