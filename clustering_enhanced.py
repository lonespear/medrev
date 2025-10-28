"""
Enhanced Clustering Module for MedScrape
Includes lightweight embeddings (Doc2Vec, FastText) and review/non-review comparison
Optimized for performance with large document collections
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, PCA
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
        """Basic text preprocessing"""
        texts = self.df[self.text_column].fillna('')
        # Remove empty abstracts
        self.df = self.df[texts.str.len() > 0].reset_index(drop=True)
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
        # Create embeddings
        if embedding_type == 'tfidf':
            features = self.create_tfidf_features()
            if hasattr(features, 'toarray'):
                features = features.toarray()
        elif embedding_type == 'doc2vec':
            features = self.create_doc2vec_embeddings()
        elif embedding_type == 'fasttext':
            features = self.create_fasttext_embeddings()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        # Cluster
        if method == 'kmeans':
            labels = self.cluster_kmeans(features, n_clusters)
        elif method == 'dbscan':
            labels = self.cluster_dbscan(features)
        elif method == 'hierarchical':
            labels = self.cluster_hierarchical(features, n_clusters)
        elif method == 'lda':
            labels, top_words = self.topic_modeling_lda(n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        self.cluster_labels = labels
        self.df['Cluster'] = labels
        
        # Reduce dimensions for visualization
        reduced = self.reduce_dimensions(features, reduction_method)
        self.df['Component_1'] = reduced[:, 0]
        self.df['Component_2'] = reduced[:, 1]
        
        # Get cluster summaries
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
        Generate summaries for each cluster using TF-IDF
        
        Returns:
            Dictionary mapping cluster ID to summary info
        """
        summaries = {}
        
        for cluster_id in self.df['Cluster'].unique():
            if cluster_id == -1:  # Skip noise cluster from DBSCAN
                continue
            
            cluster_docs = self.df[self.df['Cluster'] == cluster_id]
            
            # Combine abstracts in cluster
            combined_text = ' '.join(cluster_docs[self.text_column].fillna(''))
            
            # Get top TF-IDF terms
            vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')
            try:
                tfidf = vectorizer.fit_transform([combined_text])
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf.toarray()[0]
                top_terms = [feature_names[i] for i in scores.argsort()[-top_n:][::-1]]
            except:
                top_terms = []
            
            summaries[cluster_id] = {
                'size': len(cluster_docs),
                'top_terms': top_terms,
                'sample_titles': cluster_docs['Title'].head(3).tolist() if 'Title' in cluster_docs.columns else []
            }
        
        return summaries


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
