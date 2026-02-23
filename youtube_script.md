# MedScrape YouTube Demo Script
### Department of Mathematical Sciences, USMA

---

## 1. Hook

- Systematic literature reviews are the backbone of evidence-based medicine — but manually reading thousands of papers isn't feasible for any research team. This tool automates that entire process.
- MedScrape was developed at the Department of Mathematical Sciences at West Point, and the methodology behind it has been accepted for publication. Today I'm going to show you exactly what it does.

---

## 2. Data Collection

- The first thing you'll do is tell the app where to get its data. You have two options: scrape PubMed directly, or upload a CSV you've already collected.
- (Open the sidebar. Point to the Analysis Mode selector and the Data Source toggle.)
- The query builder lets you construct complex Boolean searches — keywords within a group are joined by OR, and groups can be combined with AND, OR, or NOT. The full query previews live as you type so you always know exactly what's being sent to PubMed.
- (Switch to CSV upload. Load the AI & Nutrition dataset from vignettes/data/.)
- Once the data loads, you get an instant summary — total articles, year range, unique journals — and every paper is automatically tagged by type: reviews, systematic reviews, clinical trials, RCTs, meta-analyses, and longitudinal studies.
- (Point to the article type breakdown panel.)
- The publication trends chart shows you how the field has grown over time. Notice the trend line — this field has seen significant growth, which is exactly the kind of context that belongs in the introduction of a systematic review.
- (Point to the trend chart and growth annotation.)

---

## 3. Clustering

- Now we move to the core of the analysis. The app uses unsupervised machine learning to group thousands of papers by theme — no human labeling required.
- (Click the Clustering tab. In the settings panel, select TF-IDF for embedding, K-Means for clustering, PCA for visualization, and set clusters to 8. Click Run Clustering.)
- In under ten seconds, the model has read every abstract and organized the literature into thematic clusters. Each dot here is a paper, and papers near each other are about similar topics.
- (Point to the interactive scatter plot. Hover over a few points.)
- Let's look at what the model actually found. Expand a few of these cluster summaries and read the top terms.
- (Expand 2–3 cluster summary panels.)
- The model found these clusters entirely on its own — no topic list was given to it. This is what the literature actually looks like when you let the math do the work.

---

## 4. Gap Analysis

- This is the feature that forms the scientific contribution of the paper. The question we're asking is: where does research exist that reviews haven't covered yet?
- (Switch Analysis Mode in the sidebar to Review vs Non-Review Comparison. Load the Microbiome search — reviews CSV and non-reviews CSV for search 4.)
- (Click the Gap Analysis tab. Click Identify Research Gaps.)
- This heatmap shows cosine similarity between research clusters and review clusters. Green means a research area is well-represented in the existing review literature. Red means the research exists — but no one has written a review on it yet. Those red cells are your next systematic review topics.
- (Point to red cells in the heatmap. Read the cluster label of one gap aloud.)
- The gap report below quantifies this — cluster size, coverage score, and the top terms defining that under-reviewed area.
- (Scroll to the gap report table.)

---

## 5. Export

- Everything the app produces is downloadable. Clustered articles, cluster summaries, the gap analysis report, and the full similarity matrix all export as CSVs.
- (Click the Export tab. Point to the four download buttons.)
- These outputs plug directly into Excel, R, Python — whatever your team already uses for downstream analysis.

---

## 6. Wrap-Up

- MedScrape brings together PubMed scraping, NLP embeddings, four clustering algorithms, and automated gap detection into a single tool that runs in a browser with no setup required.
- The paper behind this methodology is currently in revision. A link to the live app and the GitHub repository are in the description.
- (Show the app URL: medreview.streamlit.app.)
