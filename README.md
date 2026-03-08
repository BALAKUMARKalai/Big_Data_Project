# Big data project 2026 (MovieLens 100k)
 

Knowledge Base Enrichment, Data Extraction, and Dimensionality Reduction using the MovieLens 100k dataset enriched with TMDB API.

## Project Structure

```
├── data/
│   ├── movies.csv              # Original MovieLens dataset
│   ├── ratings.csv             # 100 836 user ratings
│   ├── tags.csv                # User-generated tags
│   ├── df_final.csv            # Enriched dataset (TMDB features)
│   ├── df_tsne.csv             # t-SNE 2D projection
│   ├── df_pca.csv              # PCA 2D projection
│   └── supergenres.csv         # Super-genre assignments
│
└── scripts/
    ├── 01_data_enrichment.ipynb       # TMDB API enrichment
    ├── 02_data_extraction.ipynb       # Data mining & graph analysis
    ├── 03_dimension_reduction.ipynb   # PCA, t-SNE & graph summarization
    └── 04_recommandation_system.ipynb # Movie recommendation system
```

## Pipeline

**1. Data Enrichment** :  MovieLens enriched with TMDB API (budget, revenue, runtime, vote_average) using `movieId` as foreign key. Ratings aggregated per movie (mean, std, count). Tags concatenated per movie for TF-IDF vectorization.

**2. Data Extraction & Analysis** :  FP-Growth association rules on genres (min_support=0.01). Lift matrix computed between all genre pairs. Hierarchical clustering (Ward linkage) on distance matrix `d = 1/lift` to create 3 super-genres validated by genre co-occurrence graph (Greedy Modularity communities).

**3. Dimensionality Reduction** :  PCA and t-SNE applied on feature matrix `X ∈ R^(9742×109)` (9 numerical + 100 TF-IDF features). PCA explains 95% of variance with 7 components. t-SNE Silhouette Score (0.12) outperforms PCA (-0.379), confirming non-linear genre structure.

**4. Graph Summarization** :  RDF quotient graph compressing 9742 film nodes into 3 super-nodes (41 588 → 29 triples, 99.9% reduction). Equivalence relation: two films are equivalent if they share the same super-genre. SPARQL queries on the summary graph.

## Super-genres

| Super-genre | Genres |
|-------------|--------|
| Famille | Adventure, Animation, Children, Fantasy, IMAX |
| Sombre/Suspense | Action, Crime, Film-Noir, Horror, Mystery, Sci-Fi, Thriller |
| Drame/Autres | Comedy, Documentary, Drama, Musical, Romance, War, Western |

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
pip install mlxtend scipy networkx rdflib
```

## Dataset

MovieLens 100k — [GroupLens](https://grouplens.org/datasets/movielens/)  
TMDB API — [themoviedb.org](https://www.themoviedb.org/)
