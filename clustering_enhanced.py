"""
Enhanced Clustering Module for MedScrape
Includes lightweight embeddings (Doc2Vec, FastText) and review/non-review comparison
Optimized for performance with large document collections
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import umap
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import FastText
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EnhancedClusterer:
    """Enhanced clustering with lightweight embeddings (TF-IDF, Doc2Vec, FastText) and review comparison"""
    
    def __init__(self, df: pd.DataFrame, text_column: str = 'Abstract'):
        """
        Initialize clusterer with dataframe
        
        Args:
            df: DataFrame with abstracts
            text_column: Column name containing text to cluster
        """
        self.df = df.copy()
        self.text_column = text_column
        self.vectorizer = None
        self.tfidf_matrix = None
        self.embeddings = None
        self.cluster_labels = None
        self.reduced_features = None
        
    def preprocess_text(self) -> pd.Series:
        """Basic text preprocessing: strip HTML tags, normalize whitespace, drop empties."""
        def clean(text):
            if not isinstance(text, str):
                return ''
            # Strip HTML/XML tags (<sup>, <sub>, <i>, etc.)
            text = re.sub(r'<[^>]+>', ' ', text)
            # Collapse whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        self.df[self.text_column] = self.df[self.text_column].fillna('').apply(clean)
        self.df = self.df[self.df[self.text_column].str.len() > 0].reset_index(drop=True)
        return self.df[self.text_column]
    
    def create_tfidf_features(self, max_features: int = 5000, 
                              ngram_range: Tuple[int, int] = (1, 2)) -> np.ndarray:
        """
        Create TF-IDF features
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range for TF-IDF
            
        Returns:
            TF-IDF matrix
        """
        texts = self.preprocess_text()
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self.tfidf_matrix
    
    def create_doc2vec_embeddings(self, vector_size: int = 100, epochs: int = 20) -> np.ndarray:
        """
        Create embeddings using Doc2Vec (lighter than transformers)

        Args:
            vector_size: Dimension of document vectors
            epochs: Number of training epochs

        Returns:
            Embedding matrix
        """
        texts = self.preprocess_text()

        # Tokenize documents
        tagged_data = [TaggedDocument(words=text.lower().split(), tags=[str(i)])
                       for i, text in enumerate(texts)]

        # Train Doc2Vec model
        model = Doc2Vec(
            vector_size=vector_size,
            min_count=2,
            epochs=epochs,
            workers=4,
            dm=1  # PV-DM (distributed memory)
        )

        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

        # Generate embeddings
        self.embeddings = np.array([model.dv[str(i)] for i in range(len(texts))])
        return self.embeddings

    def create_fasttext_embeddings(self, vector_size: int = 100, epochs: int = 10) -> np.ndarray:
        """
        Create embeddings using FastText (very lightweight and fast)

        Args:
            vector_size: Dimension of word vectors
            epochs: Number of training epochs

        Returns:
            Embedding matrix (averaged word vectors per document)
        """
        texts = self.preprocess_text()

        # Tokenize documents
        sentences = [text.lower().split() for text in texts]

        # Train FastText model
        model = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=5,
            min_count=2,
            epochs=epochs,
            workers=4
        )

        # Generate document embeddings by averaging word vectors
        embeddings = []
        for sent in sentences:
            # Get vectors for words that exist in vocabulary
            word_vecs = [model.wv[word] for word in sent if word in model.wv]
            if word_vecs:
                # Average word vectors
                embeddings.append(np.mean(word_vecs, axis=0))
            else:
                # Use zero vector if no words found
                embeddings.append(np.zeros(vector_size))

        self.embeddings = np.array(embeddings)
        return self.embeddings
    
    def cluster_kmeans(self, features: np.ndarray, n_clusters: int = 8) -> np.ndarray:
        """K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(features)
    
    def cluster_dbscan(self, features: np.ndarray, eps: float = 0.5, 
                       min_samples: int = 5) -> np.ndarray:
        """DBSCAN clustering"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        return dbscan.fit_predict(features)
    
    def cluster_hierarchical(self, features: np.ndarray, n_clusters: int = 8) -> np.ndarray:
        """Hierarchical clustering"""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, 
                                               metric='cosine', 
                                               linkage='average')
        return hierarchical.fit_predict(features)
    
    def topic_modeling_lda(self, n_topics: int = 10) -> Tuple[np.ndarray, List[List[str]]]:
        """
        LDA topic modeling
        
        Returns:
            Topic assignments and top words per topic
        """
        if self.tfidf_matrix is None:
            self.create_tfidf_features()
        
        lda = LatentDirichletAllocation(n_components=n_topics, 
                                        random_state=42,
                                        max_iter=20)
        topic_assignments = lda.fit_transform(self.tfidf_matrix)
        
        # Get top words per topic
        feature_names = self.vectorizer.get_feature_names_out()
        top_words = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words.append([feature_names[i] for i in top_indices])
        
        return topic_assignments.argmax(axis=1), top_words
    
    def reduce_dimensions(self, features: np.ndarray, method: str = 'pca', 
                         n_components: int = 2) -> np.ndarray:
        """
        Dimensionality reduction
        
        Args:
            features: Feature matrix
            method: 'pca', 'tsne', or 'umap'
            n_components: Number of dimensions
            
        Returns:
            Reduced features
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, 
                          perplexity=30, n_iter=1000)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42,
                               n_neighbors=15, min_dist=0.1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.reduced_features = reducer.fit_transform(features)
        return self.reduced_features
    
    def cluster_and_reduce(self, method: str = 'kmeans', n_clusters: int = 8,
                          embedding_type: str = 'tfidf',
                          reduction_method: str = 'pca') -> Dict:
        """
        Complete pipeline: embed, cluster, and reduce

        Args:
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical', 'lda')
            n_clusters: Number of clusters (for applicable methods)
            embedding_type: 'tfidf', 'doc2vec', or 'fasttext'
            reduction_method: 'pca', 'tsne', or 'umap'

        Returns:
            Dictionary with results
        """
        # LDA uses TF-IDF internally — handle separately
        if method == 'lda':
            sparse = self.create_tfidf_features()
            labels, top_words = self.topic_modeling_lda(n_clusters)
            # Reduce the sparse TF-IDF for visualization via TruncatedSVD
            n_svd = min(50, sparse.shape[1] - 1)
            features = TruncatedSVD(n_components=n_svd, random_state=42).fit_transform(sparse)
        elif embedding_type == 'tfidf':
            sparse = self.create_tfidf_features()
            # TruncatedSVD works directly on the sparse matrix — avoids creating a
            # 392 MB dense array and cuts clustering time from ~2 min to ~15 sec.
            n_svd = min(100, sparse.shape[1] - 1)
            features = TruncatedSVD(n_components=n_svd, random_state=42).fit_transform(sparse)
            if method == 'kmeans':
                labels = self.cluster_kmeans(features, n_clusters)
            elif method == 'dbscan':
                labels = self.cluster_dbscan(features)
            elif method == 'hierarchical':
                labels = self.cluster_hierarchical(features, n_clusters)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
        elif embedding_type == 'doc2vec':
            features = self.create_doc2vec_embeddings()
            if method == 'kmeans':
                labels = self.cluster_kmeans(features, n_clusters)
            elif method == 'dbscan':
                labels = self.cluster_dbscan(features)
            elif method == 'hierarchical':
                labels = self.cluster_hierarchical(features, n_clusters)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
        elif embedding_type == 'fasttext':
            features = self.create_fasttext_embeddings()
            if method == 'kmeans':
                labels = self.cluster_kmeans(features, n_clusters)
            elif method == 'dbscan':
                labels = self.cluster_dbscan(features)
            elif method == 'hierarchical':
                labels = self.cluster_hierarchical(features, n_clusters)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        self.cluster_labels = labels
        self.df['Cluster'] = labels

        # Reduce to 2D for visualization (now operating on ≤100 dims, not 5000)
        reduced = self.reduce_dimensions(features, reduction_method)
        self.df['Component_1'] = reduced[:, 0]
        self.df['Component_2'] = reduced[:, 1]

        summaries = self.get_cluster_summaries()

        results = {
            'df': self.df,
            'labels': labels,
            'reduced_features': reduced,
            'summaries': summaries,
            'n_clusters': len(np.unique(labels[labels >= 0]))
        }

        if method == 'lda':
            results['top_words'] = top_words

        return results
    
    def get_cluster_summaries(self, top_n: int = 10) -> Dict[int, Dict]:
        """
        Generate cluster summaries using the global TF-IDF matrix.

        - Top terms: derived from each cluster's centroid in the global TF-IDF space,
          so inter-cluster weighting is preserved (not just per-cluster TF-IDF).
        - Representative papers: the 5 papers with highest cosine similarity to centroid
          (not arbitrary head() order).
        - Auto-label: top bigram from centroid terms, or top two unigrams joined.
        - Headline: a representative sentence extracted from the most central paper.
        """
        summaries = {}

        for cluster_id in sorted(self.df['Cluster'].unique()):
            if cluster_id == -1:
                continue

            cluster_mask = (self.df['Cluster'] == cluster_id).values
            cluster_docs = self.df[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            # ── Top terms via global TF-IDF centroid ──────────────────────────────
            if self.tfidf_matrix is not None and self.vectorizer is not None:
                cluster_vecs = self.tfidf_matrix[cluster_indices]
                centroid = np.asarray(cluster_vecs.mean(axis=0)).flatten()
                feature_names = self.vectorizer.get_feature_names_out()
                top_indices = centroid.argsort()[-top_n:][::-1]
                top_terms = [feature_names[i] for i in top_indices if centroid[i] > 0]

                # ── Most representative papers: cosine sim to centroid ─────────────
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
                sims = np.array([
                    np.dot(
                        np.asarray(self.tfidf_matrix[i].todense()).flatten()
                        / (np.linalg.norm(np.asarray(self.tfidf_matrix[i].todense()).flatten()) + 1e-10),
                        centroid_norm
                    )
                    for i in cluster_indices
                ])
                top_local = sims.argsort()[-5:][::-1]
                rep_papers = cluster_docs.iloc[top_local]
            else:
                # Fallback when TF-IDF matrix isn't available (Doc2Vec / FastText path)
                combined = ' '.join(cluster_docs[self.text_column].fillna(''))
                try:
                    v = TfidfVectorizer(max_features=top_n, stop_words='english')
                    mat = v.fit_transform([combined])
                    fn = v.get_feature_names_out()
                    sc = mat.toarray()[0]
                    top_terms = [fn[i] for i in sc.argsort()[-top_n:][::-1]]
                except Exception:
                    top_terms = []
                rep_papers = cluster_docs.head(5)

            label = self._auto_label(top_terms)
            headline = self._extract_snippet(
                rep_papers[self.text_column].iloc[0] if len(rep_papers) > 0 else '',
                top_terms
            )

            summaries[cluster_id] = {
                'size': len(cluster_docs),
                'label': label,
                'top_terms': top_terms,
                'headline': headline,
                'sample_titles': rep_papers['Title'].tolist() if 'Title' in rep_papers.columns else [],
                'sample_urls': rep_papers['URL'].tolist() if 'URL' in rep_papers.columns else [],
            }

        return summaries

    def _auto_label(self, top_terms: List[str]) -> str:
        """Generate a short cluster label: prefer the top bigram, else join top two unigrams."""
        bigrams = [t for t in top_terms if ' ' in t]
        if bigrams:
            return bigrams[0].title()
        if len(top_terms) >= 2:
            return f"{top_terms[0].title()} & {top_terms[1].title()}"
        return top_terms[0].title() if top_terms else "Unlabeled"

    def _extract_snippet(self, abstract: str, top_terms: List[str]) -> str:
        """
        Return up to 2 representative sentences from the abstract that mention
        at least one top term.  Falls back to the first 2 substantial sentences.
        """
        if not abstract:
            return ''
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', abstract) if len(s.strip()) >= 40]
        terms_lower = {t.lower() for t in top_terms[:6]}
        good = [s for s in sentences if any(t in s.lower() for t in terms_lower)]
        result = good[:2] if good else sentences[:2]
        return ' '.join(result)


class ReviewComparator:
    """Compare review and non-review papers to identify gaps"""
    
    def __init__(self, review_df: pd.DataFrame, non_review_df: pd.DataFrame,
                 text_column: str = 'Abstract'):
        """
        Initialize comparator
        
        Args:
            review_df: DataFrame with review articles
            non_review_df: DataFrame with non-review articles
            text_column: Column containing text
        """
        self.review_df = review_df.copy()
        self.non_review_df = non_review_df.copy()
        self.text_column = text_column
        
    def cluster_both(self, method: str = 'kmeans', n_clusters: int = 8,
                    embedding_type: str = 'tfidf') -> Tuple[Dict, Dict]:
        """
        Cluster both review and non-review papers
        
        Returns:
            Results for review and non-review papers
        """
        # Cluster reviews
        review_clusterer = EnhancedClusterer(self.review_df, self.text_column)
        review_results = review_clusterer.cluster_and_reduce(
            method=method, n_clusters=n_clusters, 
            embedding_type=embedding_type, reduction_method='pca'
        )
        
        # Cluster non-reviews
        non_review_clusterer = EnhancedClusterer(self.non_review_df, self.text_column)
        non_review_results = non_review_clusterer.cluster_and_reduce(
            method=method, n_clusters=n_clusters,
            embedding_type=embedding_type, reduction_method='pca'
        )
        
        self.review_results = review_results
        self.non_review_results = non_review_results
        
        return review_results, non_review_results
    
    def compute_cosine_similarity_matrix(self, review_results: Dict, 
                                        non_review_results: Dict) -> np.ndarray:
        """
        Compute pairwise cosine similarity between review and non-review clusters
        
        Returns:
            Cosine similarity matrix (non-review clusters x review clusters)
        """
        # Create cluster documents
        review_cluster_docs = []
        non_review_cluster_docs = []
        
        # Aggregate abstracts by cluster for reviews
        for cluster_id in sorted(review_results['df']['Cluster'].unique()):
            if cluster_id >= 0:
                texts = review_results['df'][review_results['df']['Cluster'] == cluster_id][self.text_column]
                review_cluster_docs.append(' '.join(texts.fillna('')))
        
        # Aggregate abstracts by cluster for non-reviews
        for cluster_id in sorted(non_review_results['df']['Cluster'].unique()):
            if cluster_id >= 0:
                texts = non_review_results['df'][non_review_results['df']['Cluster'] == cluster_id][self.text_column]
                non_review_cluster_docs.append(' '.join(texts.fillna('')))
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        all_docs = review_cluster_docs + non_review_cluster_docs
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        
        # Split back
        n_review = len(review_cluster_docs)
        review_vectors = tfidf_matrix[:n_review]
        non_review_vectors = tfidf_matrix[n_review:]
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(non_review_vectors, review_vectors)
        
        return similarity_matrix
    
    def identify_gaps(self, similarity_matrix: np.ndarray, 
                     threshold: float = 0.3) -> List[int]:
        """
        Identify non-review clusters poorly covered by reviews
        
        Args:
            similarity_matrix: Cosine similarity matrix
            threshold: Similarity threshold below which a gap is identified
            
        Returns:
            List of non-review cluster IDs representing gaps
        """
        # For each non-review cluster, find max similarity to any review cluster
        max_similarities = similarity_matrix.max(axis=1)
        
        # Identify gaps (low similarity to all review clusters)
        gap_indices = np.where(max_similarities < threshold)[0]
        
        return gap_indices.tolist()
    
    def create_similarity_heatmap(self, similarity_matrix: np.ndarray,
                                 review_labels: List[str] = None,
                                 non_review_labels: List[str] = None) -> go.Figure:
        """
        Create interactive heatmap of cosine similarities
        
        Args:
            similarity_matrix: Cosine similarity matrix
            review_labels: Labels for review clusters
            non_review_labels: Labels for non-review clusters
            
        Returns:
            Plotly figure
        """
        if review_labels is None:
            review_labels = [f"Review {i}" for i in range(similarity_matrix.shape[1])]
        
        if non_review_labels is None:
            non_review_labels = [f"Research {i}" for i in range(similarity_matrix.shape[0])]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=review_labels,
            y=non_review_labels,
            colorscale='RdYlGn',
            colorbar=dict(title="Cosine<br>Similarity"),
            hoverongaps=False,
            hovertemplate='%{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Review Coverage of Research Clusters',
            xaxis_title='Review Paper Clusters',
            yaxis_title='Research Paper Clusters',
            width=800,
            height=600,
            font=dict(size=12)
        )
        
        return fig
    
    def generate_gap_report(self, gap_indices: List[int], 
                           non_review_results: Dict) -> pd.DataFrame:
        """
        Generate detailed report on identified gaps
        
        Args:
            gap_indices: Cluster IDs representing gaps
            non_review_results: Clustering results for non-review papers
            
        Returns:
            DataFrame with gap analysis
        """
        gap_data = []
        
        for idx in gap_indices:
            cluster_df = non_review_results['df'][non_review_results['df']['Cluster'] == idx]
            summary = non_review_results['summaries'].get(idx, {})
            
            gap_data.append({
                'Cluster_ID': idx,
                'Cluster_Size': summary.get('size', 0),
                'Top_Terms': ', '.join(summary.get('top_terms', [])[:5]),
                'Sample_Papers': len(summary.get('sample_titles', [])),
                'Avg_Year': cluster_df['Year'].mean() if 'Year' in cluster_df.columns else None
            })
        
        return pd.DataFrame(gap_data)


def create_interactive_plot(df: pd.DataFrame, color_by: str = 'Cluster',
                           hover_data: List[str] = None) -> go.Figure:
    """
    Create interactive scatter plot of clusters
    
    Args:
        df: DataFrame with Component_1, Component_2, and cluster assignments
        color_by: Column to color by
        hover_data: Additional columns to show on hover
        
    Returns:
        Plotly figure
    """
    if hover_data is None:
        hover_data = ['Title', 'Year', 'Journal'] if all(col in df.columns for col in ['Title', 'Year', 'Journal']) else []
    
    fig = px.scatter(
        df,
        x='Component_1',
        y='Component_2',
        color=color_by,
        hover_data=hover_data,
        title='Article Clusters',
        labels={'Component_1': 'Component 1', 'Component_2': 'Component 2'}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        width=900,
        height=700,
        hovermode='closest'
    )
    
    return fig
