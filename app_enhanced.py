"""
MedScrape: Enhanced Literature Review Application
Streamlit app for automated PubMed literature analysis with AI-assisted gap identification
"""

import streamlit as st
import pandas as pd
import numpy as np
from Bio import Entrez
import time
from datetime import datetime
import io
import re
from streamlit_tags import st_tags
from clustering_enhanced import EnhancedClusterer, ReviewComparator, create_interactive_plot

# Page configuration
st.set_page_config(
    page_title="MedScrape: AI Literature Review",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    /* Make sidebar wider */
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 450px;
        max-width: 450px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 450px;
        max-width: 450px;
        margin-left: -450px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = None
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None
if 'review_data' not in st.session_state:
    st.session_state.review_data = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'query_groups' not in st.session_state:
    st.session_state.query_groups = [{'terms': [], 'operator': 'AND'}]


def scrape_pubmed(query: str, email: str, max_results: int = 10000, 
                  start_year: int = None, end_year: int = None,
                  publication_type: str = "all") -> pd.DataFrame:
    """
    Scrape PubMed database
    
    Args:
        query: Search query
        email: Email for NCBI
        max_results: Maximum results per query
        start_year: Start year for filtering
        end_year: End year for filtering
        publication_type: 'all', 'review', or 'non-review'
        
    Returns:
        DataFrame with results
    """
    Entrez.email = email
    
    # Add publication type filter
    if publication_type == "review":
        query += ' AND "review"[Publication Type]'
    elif publication_type == "non-review":
        query += ' NOT "review"[Publication Type]'
    
    # Add date range if specified
    if start_year and end_year:
        query += f' AND {start_year}:{end_year}[pdat]'
    
    # Search
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        id_list = record["IdList"]
        handle.close()
        
        if not id_list:
            return pd.DataFrame()
        
        # Fetch details in batches
        batch_size = 500
        all_records = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i + batch_size]
            
            status_text.text(f"Fetching records {i+1}-{min(i+batch_size, len(id_list))} of {len(id_list)}...")
            progress_bar.progress(min((i + batch_size) / len(id_list), 1.0))
            
            handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            for record in records['PubmedArticle']:
                try:
                    article = record['MedlineCitation']['Article']

                    # Extract authors
                    authors = []
                    if 'AuthorList' in article:
                        for author in article['AuthorList']:
                            if 'LastName' in author and 'Initials' in author:
                                authors.append(f"{author['LastName']} {author['Initials']}")

                    # Extract abstract
                    abstract = ''
                    if 'Abstract' in article:
                        abstract_parts = article['Abstract'].get('AbstractText', [])
                        if abstract_parts:
                            abstract = ' '.join([str(part) for part in abstract_parts])

                    # Extract year
                    year = None
                    if 'ArticleDate' in article and article['ArticleDate']:
                        year = int(article['ArticleDate'][0]['Year'])
                    elif 'Journal' in article and 'JournalIssue' in article['Journal']:
                        pub_date = article['Journal']['JournalIssue'].get('PubDate', {})
                        if 'Year' in pub_date:
                            year = int(pub_date['Year'])

                    # Identify article types from publication type list
                    article_type_list = article.get('PublicationTypeList', [])
                    is_review = any(pub_type.lower() == "review" for pub_type in article_type_list)
                    is_sys_rev = any(pub_type.lower() == "systematic review" for pub_type in article_type_list)
                    is_clinical_trial = any(pub_type.lower() == "clinical trial" for pub_type in article_type_list)
                    is_meta_analysis = any(pub_type.lower() == "meta-analysis" for pub_type in article_type_list)
                    is_rct = any(pub_type.lower() == "randomized controlled trial" for pub_type in article_type_list)

                    # Detect longitudinal studies from abstract keywords
                    longitudinal_terms = ["longitudinal", "long-term follow up", "long term follow up",
                                         "follow-up", "follow up", "prospective cohort"]
                    is_longitudinal = any(term in abstract.lower() for term in longitudinal_terms)

                    all_records.append({
                        'PMID': record['MedlineCitation']['PMID'],
                        'Title': article.get('ArticleTitle', ''),
                        'Abstract': abstract,
                        'Authors': '; '.join(authors),
                        'Journal': article.get('Journal', {}).get('Title', ''),
                        'Year': year,
                        'URL': f"https://pubmed.ncbi.nlm.nih.gov/{record['MedlineCitation']['PMID']}/",
                        'Review': 1 if is_review else 0,
                        'SystematicReview': 1 if is_sys_rev else 0,
                        'ClinicalTrial': 1 if is_clinical_trial else 0,
                        'MetaAnalysis': 1 if is_meta_analysis else 0,
                        'RCT': 1 if is_rct else 0,
                        'LongitudinalStudy': 1 if is_longitudinal else 0
                    })
                except Exception as e:
                    continue
            
            time.sleep(0.5)  # Rate limiting
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(all_records)
    
    except Exception as e:
        st.error(f"Error scraping PubMed: {str(e)}")
        return pd.DataFrame()


def export_results(df: pd.DataFrame, filename: str = "results.csv") -> bytes:
    """Export DataFrame to CSV"""
    return df.to_csv(index=False).encode('utf-8')


def build_query_from_groups(query_groups):
    """
    Build PubMed query string from query groups.

    Args:
        query_groups: List of dicts with 'terms' and 'operator' keys

    Returns:
        Query string
    """
    if not query_groups or all(not group['terms'] for group in query_groups):
        return ""

    query_parts = []
    for i, group in enumerate(query_groups):
        if not group['terms']:
            continue

        # Build group query (OR all terms within the group)
        group_query = " OR ".join([f'"{term}"' for term in group['terms']])

        # Wrap in parentheses
        if len(group['terms']) > 1:
            group_query = f"({group_query})"
        else:
            group_query = f'"{group["terms"][0]}"'

        # Add operator prefix for groups after the first
        if i == 0:
            query_parts.append(group_query)
        else:
            operator = group['operator']
            if operator == "NOT":
                query_parts.append(f"NOT {group_query}")
            else:
                query_parts.append(f"{operator} {group_query}")

    return " ".join(query_parts)


def main():
    """Main application"""
    
    # Header
    st.title("📚 MedScrape: AI-Assisted Literature Review")
    st.markdown("""
    Automated literature search and gap analysis using unsupervised learning and NLP.
    Developed by the Department of Mathematical Sciences, United States Military Academy.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Single Search", "Review vs Non-Review Comparison"],
            help="Choose between basic clustering or comparing review and research papers"
        )
        
        st.markdown("---")
        
        # Data source
        data_source = st.radio("Data Source", ["Scrape PubMed", "Upload CSV"])
        
        if data_source == "Scrape PubMed":
            st.subheader("PubMed Search")
            email = st.text_input("Email (required by NCBI)", value="user@example.com")

            st.markdown("### Query Builder")
            st.markdown("Build your query by adding keyword groups. Terms within each group are combined with OR, and groups are chained with AND/OR/NOT operators.")

            # Display and manage query groups
            for idx, group in enumerate(st.session_state.query_groups):
                st.markdown(f"**Group {idx + 1}**")

                col1, col2 = st.columns([4, 1])
                with col1:
                    # Keywords input using st_tags
                    group['terms'] = st_tags(
                        label=f"Keywords for Group {idx + 1}",
                        text="Press enter to add more",
                        value=group['terms'],
                        key=f'group_{idx}_terms'
                    )

                with col2:
                    if idx > 0:
                        # Operator selection for groups after the first
                        group['operator'] = st.selectbox(
                            "Operator",
                            options=["AND", "OR", "NOT"],
                            index=["AND", "OR", "NOT"].index(group['operator']),
                            key=f'group_{idx}_operator'
                        )
                    else:
                        st.write("")  # Spacer

                # Remove group button
                if len(st.session_state.query_groups) > 1:
                    if st.button(f"Remove Group {idx + 1}", key=f'remove_group_{idx}'):
                        st.session_state.query_groups.pop(idx)
                        st.rerun()

            # Add group button
            if st.button("➕ Add Group"):
                st.session_state.query_groups.append({'terms': [], 'operator': 'AND'})
                st.rerun()

            # Build and display final query
            query = build_query_from_groups(st.session_state.query_groups)
            st.markdown("**Final Query:**")
            st.code(query if query else "(empty query)")

            col1, col2 = st.columns(2)
            with col1:
                start_year = st.number_input("Start Year", min_value=1900, max_value=2025, value=2000)
            with col2:
                end_year = st.number_input("End Year", min_value=1900, max_value=2025, value=2025)

            max_results = st.number_input("Max Results", min_value=100, max_value=50000, value=10000)

            if analysis_mode == "Review vs Non-Review Comparison":
                st.info("Will scrape both review and non-review papers")
        else:
            st.subheader("Upload Data")
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            st.markdown("""
            **Required columns:** Title, Abstract, Authors, Journal, Year
            """)
        
        st.markdown("---")
        
        # Clustering parameters
        st.subheader("🧠 Clustering Settings")
        
        embedding_type = st.selectbox(
            "Embedding Method",
            ["TF-IDF", "Doc2Vec", "FastText"],
            help="""
            • TF-IDF: Fastest, best for keyword-based clustering (recommended)
            • Doc2Vec: Captures document semantics, moderate speed
            • FastText: Lightweight word embeddings, good balance
            """
        )
        
        clustering_method = st.selectbox(
            "Clustering Algorithm",
            ["K-Means", "DBSCAN", "Hierarchical", "LDA"],
            help="Choose clustering algorithm"
        )
        
        if clustering_method in ["K-Means", "Hierarchical", "LDA"]:
            n_clusters = st.slider("Number of Clusters", min_value=3, max_value=20, value=8)
        
        reduction_method = st.selectbox(
            "Visualization Method",
            ["PCA", "t-SNE", "UMAP"],
            help="Dimensionality reduction for visualization"
        )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Data Collection", "🔍 Clustering", "🎯 Gap Analysis", "📊 Export"])
    
    # Tab 1: Data Collection
    with tab1:
        st.header("Data Collection")
        
        if data_source == "Scrape PubMed":
            if st.button("🔍 Start PubMed Scrape", type="primary"):
                with st.spinner("Scraping PubMed database..."):
                    if analysis_mode == "Single Search":
                        df = scrape_pubmed(query, email, max_results, start_year, end_year, "all")
                        if not df.empty:
                            st.session_state.scraped_data = df
                            st.success(f"✅ Successfully scraped {len(df)} articles!")
                    else:  # Review comparison
                        # Scrape reviews
                        st.info("Scraping review papers...")
                        review_df = scrape_pubmed(query, email, max_results, start_year, end_year, "review")
                        
                        # Scrape non-reviews
                        st.info("Scraping research papers...")
                        non_review_df = scrape_pubmed(query, email, max_results, start_year, end_year, "non-review")
                        
                        if not review_df.empty and not non_review_df.empty:
                            st.session_state.scraped_data = non_review_df
                            st.session_state.review_data = review_df
                            st.success(f"✅ Scraped {len(review_df)} reviews and {len(non_review_df)} research papers!")
        else:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                required_cols = ['Title', 'Abstract', 'Authors', 'Journal', 'Year']
                
                if all(col in df.columns for col in required_cols):
                    st.session_state.scraped_data = df
                    st.success(f"✅ Loaded {len(df)} articles from CSV!")
                else:
                    st.error(f"Missing required columns: {[c for c in required_cols if c not in df.columns]}")
        
        # Display data
        if st.session_state.scraped_data is not None:
            st.subheader("Preview")
            df = st.session_state.scraped_data

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Articles", len(df))
            col2.metric("Year Range", f"{df['Year'].min():.0f} - {df['Year'].max():.0f}")
            col3.metric("Unique Journals", df['Journal'].nunique())

            # Article type distribution
            st.subheader("Article Type Distribution")
            article_type_cols = ['Review', 'SystematicReview', 'ClinicalTrial',
                                'MetaAnalysis', 'RCT', 'LongitudinalStudy']

            if all(col in df.columns for col in article_type_cols):
                type_counts = {col: df[col].sum() for col in article_type_cols}
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Reviews", int(type_counts['Review']))
                col2.metric("Systematic Reviews", int(type_counts['SystematicReview']))
                col3.metric("Clinical Trials", int(type_counts['ClinicalTrial']))
                col4.metric("Meta-Analyses", int(type_counts['MetaAnalysis']))
                col5.metric("RCTs", int(type_counts['RCT']))
                col6.metric("Longitudinal", int(type_counts['LongitudinalStudy']))

            st.dataframe(df.head(100), use_container_width=True)

            # Year distribution
            st.subheader("Publication Trends")
            year_counts = df['Year'].value_counts().sort_index()
            st.bar_chart(year_counts)
            
        if st.session_state.review_data is not None:
            st.subheader("Review Papers Preview")
            st.dataframe(st.session_state.review_data.head(50), use_container_width=True)
    
    # Tab 2: Clustering
    with tab2:
        st.header("Clustering Analysis")
        
        if st.session_state.scraped_data is not None:
            if st.button("🚀 Run Clustering", type="primary"):
                with st.spinner("Running clustering analysis..."):
                    df = st.session_state.scraped_data
                    
                    # Initialize clusterer
                    clusterer = EnhancedClusterer(df, text_column='Abstract')
                    
                    # Run clustering
                    embed_type_map = {
                        "TF-IDF": "tfidf",
                        "Doc2Vec": "doc2vec",
                        "FastText": "fasttext"
                    }
                    embed_type = embed_type_map[embedding_type]
                    method_map = {
                        "K-Means": "kmeans",
                        "DBSCAN": "dbscan",
                        "Hierarchical": "hierarchical",
                        "LDA": "lda"
                    }
                    method = method_map[clustering_method]
                    
                    n_clust = n_clusters if clustering_method in ["K-Means", "Hierarchical", "LDA"] else 8
                    
                    results = clusterer.cluster_and_reduce(
                        method=method,
                        n_clusters=n_clust,
                        embedding_type=embed_type,
                        reduction_method=reduction_method.lower()
                    )
                    
                    st.session_state.clustering_results = results
                    st.success("✅ Clustering complete!")
            
            # Display results
            if st.session_state.clustering_results is not None:
                results = st.session_state.clustering_results
                
                st.subheader("Cluster Overview")
                st.metric("Number of Clusters", results['n_clusters'])
                
                # Interactive plot
                st.subheader("Interactive Visualization")
                fig = create_interactive_plot(
                    results['df'],
                    color_by='Cluster',
                    hover_data=['Title', 'Year', 'Journal']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster summaries
                st.subheader("Cluster Summaries")
                for cluster_id, summary in results['summaries'].items():
                    with st.expander(f"Cluster {cluster_id} ({summary['size']} papers)"):
                        st.markdown(f"**Top Terms:** {', '.join(summary['top_terms'][:10])}")
                        st.markdown("**Sample Titles:**")
                        for title in summary['sample_titles']:
                            st.markdown(f"- {title}")
                
                # LDA topics
                if 'top_words' in results:
                    st.subheader("LDA Topics")
                    for i, words in enumerate(results['top_words']):
                        st.markdown(f"**Topic {i}:** {', '.join(words)}")
        else:
            st.info("👈 Please collect data first in the Data Collection tab")
    
    # Tab 3: Gap Analysis
    with tab3:
        st.header("Gap Analysis: Review vs Research Papers")
        
        if analysis_mode == "Single Search":
            st.info("Gap analysis is only available in 'Review vs Non-Review Comparison' mode")
        elif st.session_state.scraped_data is not None and st.session_state.review_data is not None:
            if st.button("🎯 Identify Research Gaps", type="primary"):
                with st.spinner("Comparing review and research papers..."):
                    # Initialize comparator
                    comparator = ReviewComparator(
                        st.session_state.review_data,
                        st.session_state.scraped_data,
                        text_column='Abstract'
                    )
                    
                    # Cluster both
                    embed_type_map = {
                        "TF-IDF": "tfidf",
                        "Doc2Vec": "doc2vec",
                        "FastText": "fasttext"
                    }
                    embed_type = embed_type_map[embedding_type]
                    method_map = {
                        "K-Means": "kmeans",
                        "DBSCAN": "dbscan",
                        "Hierarchical": "hierarchical",
                        "LDA": "lda"
                    }
                    method = method_map[clustering_method]
                    n_clust = n_clusters if clustering_method in ["K-Means", "Hierarchical", "LDA"] else 8
                    
                    review_results, non_review_results = comparator.cluster_both(
                        method=method,
                        n_clusters=n_clust,
                        embedding_type=embed_type
                    )
                    
                    # Compute similarity
                    similarity_matrix = comparator.compute_cosine_similarity_matrix(
                        review_results, non_review_results
                    )
                    
                    # Identify gaps
                    gap_indices = comparator.identify_gaps(similarity_matrix, threshold=0.3)
                    
                    # Store results
                    st.session_state.comparison_results = {
                        'review_results': review_results,
                        'non_review_results': non_review_results,
                        'similarity_matrix': similarity_matrix,
                        'gap_indices': gap_indices,
                        'comparator': comparator
                    }
                    
                    st.success("✅ Gap analysis complete!")
            
            # Display results
            if st.session_state.comparison_results is not None:
                comp_results = st.session_state.comparison_results
                
                st.subheader("Coverage Heatmap")
                st.markdown("""
                This heatmap shows how well review papers cover each research cluster.
                **Low values (red)** indicate research gaps poorly covered by existing reviews.
                """)
                
                # Create heatmap
                fig = comp_results['comparator'].create_similarity_heatmap(
                    comp_results['similarity_matrix']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Gap report
                st.subheader("Identified Research Gaps")
                if comp_results['gap_indices']:
                    gap_report = comp_results['comparator'].generate_gap_report(
                        comp_results['gap_indices'],
                        comp_results['non_review_results']
                    )
                    st.dataframe(gap_report, use_container_width=True)
                    
                    # Detailed gap analysis
                    for idx in comp_results['gap_indices']:
                        with st.expander(f"Gap Details: Research Cluster {idx}"):
                            cluster_df = comp_results['non_review_results']['df'][
                                comp_results['non_review_results']['df']['Cluster'] == idx
                            ]
                            st.metric("Papers in Cluster", len(cluster_df))
                            st.markdown("**Top Papers:**")
                            for _, row in cluster_df.head(5).iterrows():
                                st.markdown(f"- [{row['Title']}]({row['URL']})")
                else:
                    st.success("No significant gaps identified! Reviews cover research well.")
        else:
            st.info("👈 Please scrape both review and research papers first")
    
    # Tab 4: Export
    with tab4:
        st.header("Export Results")
        
        if st.session_state.clustering_results is not None:
            st.subheader("Download Clustered Data")
            
            df_export = st.session_state.clustering_results['df']
            csv = export_results(df_export, "clustered_articles.csv")
            
            st.download_button(
                label="📥 Download Clustered Articles (CSV)",
                data=csv,
                file_name=f"clustered_articles_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Cluster summaries
            summaries_df = pd.DataFrame([
                {
                    'Cluster': k,
                    'Size': v['size'],
                    'Top_Terms': ', '.join(v['top_terms'][:5])
                }
                for k, v in st.session_state.clustering_results['summaries'].items()
            ])
            
            csv_summaries = export_results(summaries_df, "cluster_summaries.csv")
            st.download_button(
                label="📥 Download Cluster Summaries (CSV)",
                data=csv_summaries,
                file_name=f"cluster_summaries_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        if st.session_state.comparison_results is not None:
            st.subheader("Download Gap Analysis")
            
            gap_report = st.session_state.comparison_results['comparator'].generate_gap_report(
                st.session_state.comparison_results['gap_indices'],
                st.session_state.comparison_results['non_review_results']
            )
            
            csv_gaps = export_results(gap_report, "research_gaps.csv")
            st.download_button(
                label="📥 Download Gap Analysis (CSV)",
                data=csv_gaps,
                file_name=f"research_gaps_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Similarity matrix
            sim_df = pd.DataFrame(
                st.session_state.comparison_results['similarity_matrix'],
                columns=[f"Review_{i}" for i in range(st.session_state.comparison_results['similarity_matrix'].shape[1])],
                index=[f"Research_{i}" for i in range(st.session_state.comparison_results['similarity_matrix'].shape[0])]
            )
            
            csv_sim = export_results(sim_df, "similarity_matrix.csv")
            st.download_button(
                label="📥 Download Similarity Matrix (CSV)",
                data=csv_sim,
                file_name=f"similarity_matrix_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Developed by the Department of Mathematical Sciences, United States Military Academy</p>
        <p>📧 Contact: diana.thomas@westpoint.edu | 🔗 <a href='https://github.com/lonespear/medscrape'>GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
