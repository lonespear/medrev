import streamlit as st
from streamlit_tags import st_tags
from Bio import Entrez

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF
from sklearn.manifold import TSNE
from umap import UMAP


def search_pubmed(query, max_results=2000):
    """
    Perform a search on PubMed using the given query and return the results.
    
    :param query: The search query string.
    :param max_results: The maximum number of results to retrieve.
    :return: The search results in XML format.
    """
    # Use the esearch utility to search PubMed
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    # Read the search results
    record = Entrez.read(handle)
    # Close the handle to free up resources
    handle.close()
    
    # Extract the list of PubMed IDs (PMIDs) from the search results
    id_list = record["IdList"]
    
    # Use the efetch utility to fetch details for each PMID
    handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
    # Read the fetched records
    records = Entrez.read(handle)
    # Close the handle to free up resources
    handle.close()
    
    return records

# Update the parse_articles function
def parse_articles(records, long_study=True, rev_paper=True, sys_rev=True, clinical_trial=True, meta_analysis=True, rand_ct=True):
    """  
    :param records: The PubMed records in XML format.
    :return: A list of dictionaries containing the extracted information.
    """
    data = []
    for article in records["PubmedArticle"]:
        # Extract authors and concatenate their names, if available
        if "AuthorList" in article["MedlineCitation"]["Article"]:
            authors_list = []
            for author in article["MedlineCitation"]["Article"]["AuthorList"]:
                last_name = author.get("LastName", "")
                fore_name = author.get("ForeName", "")
                authors_list.append(last_name + " " + fore_name)
            authors = ", ".join(authors_list)
        else:
            authors = ""

        # Extract the article title
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]
        # Extract the article abstract (or use an empty string if not available)
        abstract = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [""])[0]
        # Extract the PubMed ID
        pubmed_id = article["MedlineCitation"]["PMID"]

        # Construct the PubMed URL
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"

        # Extracting publication year
        pub_year = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", "N/A")

        # Extract journal name
        journal_name = article["MedlineCitation"]["Article"]["Journal"]["Title"]

        # Flag for different publication types
        article_type_list = article["MedlineCitation"]["Article"].get("PublicationTypeList", [])
        is_review = any(pub_type.lower() == "review" for pub_type in article_type_list)
        is_sys_rev = any(pub_type.lower() == "systematic review" for pub_type in article_type_list)
        is_clinical_trial = any(pub_type.lower() == "clinical trial" for pub_type in article_type_list)
        is_meta_analysis = any(pub_type.lower() == "meta-analysis" for pub_type in article_type_list)
        is_randomized_controlled_trial = any(pub_type.lower() == "randomized controlled trial" for pub_type in article_type_list)
        longitudinal_terms = ["longitudinal", "long-term follow up", "long term follow up", "follow-up", "follow up"]
        longitudinal_study = any(term in abstract.lower() for term in longitudinal_terms)

        # Append the extracted and flagged data to the list
        data.append({
            "PublicationYear": pub_year,
            "Authors": authors,
            "Title": title,
            "Abstract": abstract,
            "JournalName": journal_name,
            "PubMedURL": url,
            "Review": 1 if is_review else 0,
            "SysRev": 1 if is_sys_rev else 0,
            "ClinicalTrial": 1 if is_clinical_trial else 0,
            "MetaAnalysis": 1 if is_meta_analysis else 0,
            "RCT": 1 if is_randomized_controlled_trial else 0,
            "LongitudinalStudy": 1 if longitudinal_study else 0
        })

        # Remove columns based on unchecked categories
        if not long_study:
            for row in data:
                row.pop("LongitudinalStudy", None)
        if not rev_paper:
            for row in data:
                row.pop("Review", None)
        if not sys_rev:
            for row in data:
                row.pop("SysRev", None)
        if not clinical_trial:
            for row in data:
                row.pop("ClinicalTrial", None)
        if not meta_analysis:
            for row in data:
                row.pop("MetaAnalysis", None)
        if not rand_ct:
            for row in data:
                row.pop("RCT", None)
    return data

def run_cluster(df, n_clusters, n_key_words, method):
    # Ensure we are using the K-Means method with PCA dimensionality reduction for this test
    if method == "K-Means" or method == "LDA":
        text_data = df['Title'].fillna('') + ' ' + df['Abstract'].fillna('')
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        X = vectorizer.fit_transform(text_data)

        if method == "K-Means":
        # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            df['Cluster'] = kmeans.labels_
        elif method == "LDA":
            lda = LatentDirichletAllocation(n_components=n_clusters, random_state=42, max_iter=10, learning_method='batch')
            df['Cluster'] = lda.fit_transform(X).argmax(axis=1)
        
        # Count the number of articles in each cluster
        cluster_sizes = df['Cluster'].value_counts()
        
        # Extract the top keywords for each cluster
        cluster_keywords = {}
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        
        for i in range(n_clusters):
            cluster_keywords[i] = [terms[ind] for ind in order_centroids[i, :n_key_words]]
        
        return cluster_sizes, cluster_keywords

    # Return empty values if the selected method and dimension reduction aren't "K-Means" and "PCA"
    return pd.Series(), {}

def visualize():
    # Reduce dimensionality for K-Means using PCA
    pca_kmeans = PCA(n_components=2)
    kmeans_data = pca_kmeans.fit_transform(tfidf_matrix.toarray())

    # Reduce dimensionality for LDA visualization using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_lda = tsne.fit_transform(lda_matrix)

    # Define a list of discrete colors, one for each cluster
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    cmap = ListedColormap(colors)

    # Create a scatter plot of the clusters
    plt.figure(figsize=(14, 8))

    # Subplot 1: K-Means
    plt.subplot(1,2,1)
    # Use the cluster labels to color the points
    plt.scatter(kmeans_data[:, 0], kmeans_data[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.6, marker='x')

    # Label the axes
    plt.title('K-Means Clustering of Abstracts (2D PCA Projection)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # Show the cluster centers on the 2D plot
    centers_2d = pca_kmeans.transform(kmeans.cluster_centers_)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=200, alpha=0.75, marker='X')

    # Create a discrete legend for the clusters
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
    labels = [f'Cluster {i}' for i in range(num_clusters)]
    plt.legend(handles, labels, title='Clusters')

    # Subplot 2: LDA
    plt.subplot(1,2,2)
    plt.scatter(tsne_lda[:, 0], tsne_lda[:, 1], c=lda_matrix.argmax(axis=1), cmap='tab10', s=50, alpha=0.6, marker='x')
    plt.title('LDA Topic Clustering Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    # Create a discrete legend for the topics
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
    labels = [f'Topic {i}' for i in range(num_topics)]
    plt.legend(handles, labels, title='Topics')

    # Show the side by side plots
    plt.tight_layout()
    plt.show()

adh_terms = [
    "Atypical ductal hyperplasia", 
    "Atypical hyperplasia of the breast", 
    "Atypical breast hyperplasia", 
    "Ductal hyperplasia with atypia", 
    "Atypical proliferation of ductal cells", 
    "Premalignant breast lesion", 
    "Atypical epithelial hyperplasia", 
    "Breast atypia", 
    "Proliferative breast lesion with atypia", 
    "Atypical intraductal hyperplasia", 
    "ADH (atypical ductal hyperplasia)", 
    "ADL", 
    "ADH", 
    "LCIS", 
    "Lobular carcinoma in situ"
    "Breast cancer"
]

breast_cancer_terms = [
    "breast cancer", "mammary carcinoma", "invasive ductal carcinoma (IDC)", 
    "invasive lobular carcinoma (ILC)", "ductal carcinoma in situ (DCIS)", 
    "lobular carcinoma in situ (LCIS)", "triple-negative breast cancer",
    "HER2-positive breast cancer", "BRCA1 mutations", "BRCA2 mutations", 
    "metastatic breast cancer", "hormone receptor-positive breast cancer", 
    "estrogen receptor-positive (ER-positive)", "progesterone receptor-positive (PR-positive)"]

# Instantiate df
# Initialize df in session state if not already present
if 'df' not in st.session_state:
    st.session_state.df = None

st.set_page_config(layout="wide")

# Title
st.title("Systematic Review Tool for PubMed")

st.write("""
         **This tool is designed to aid medical researchers in conducting comprehensive literature reviews by performing broad queries on 
         PubMed to capture a vast array of articles. By utilizing Natural Language Processing (NLP) techniques, the tool systematically 
         analyzes the retrieved articles, parsing key details such as abstracts, authors, and publication types. It then applies clustering 
         algorithms to group and categorize the articles based on thematic similarities, enabling researchers to quickly discern the body of 
         literature relevant to their research area. This systematic approach provides a structured way to explore and understand the existing 
         knowledge landscape within PubMed, enhancing the efficiency and accuracy of the literature review process.**
         """)

st.divider()


st.subheader("Scraping Tool")

included_1 = st_tags(
    value=adh_terms,
    label="#### Include words or phrases:",
    text="Press enter to add more",
    key='included1'
)

included_2 = st_tags(
    value=breast_cancer_terms,
    label="#### Include words or phrases:",
    text="Press enter to add more",
    key='included2'
)

excluded = st_tags(
    label="##### Exclude words or phrases:",
    text="Press enter to add more",
    maxtags=4,
    key='excluded'
)

query_inc1 = " OR ".join([f'"{term}"' for term in included_1])
query_inc2 = " OR ".join([f'"{term}"' for term in included_1])
query_ex = " AND ".join([f'NOT "{term}"' for term in excluded])
query = f'(({query_inc1}) " AND " ({query_inc2})) {query_ex}'

st.write('##### Final Query')
st.write(query)

c3, c4, c5, c6, _, c9= st.columns([3, 4, 4, 4, 8, 3])

with c3:
    max_results = st.number_input('Maximum Results', min_value=500, max_value=20000, value=2000, step=1)

with c4:
    long_study = st.checkbox("Longitudinal Study", value = True, key='long')
    rev_paper = st.checkbox("Review Paper", value = True, key='rev')

with c5:
    sys_rev = st.checkbox("Systematic Review", value = True, key='sys')
    clin_trial = st.checkbox("Review Paper", value = True, key='clin')

with c6:
    meta = st.checkbox("Meta Analysis", value = True, key='meta')
    rct = st.checkbox("RCT", value = True, key='rct')

with c9:
    run_query = st.button("Run Search", type='primary')

# Perform the search when the button is clicked
if run_query:
    if included_1 == []:
        st.write("Query is empty!")
        run_query = False
    else:
        placeholder = st.empty()
        placeholder.write("Searching PubMed...")
        # Provide your email for the Entrez API
        Entrez.email = "your.email@example.com"  # Replace with your email
        records = search_pubmed(query, max_results=max_results)
        
        # Parse the search results
        articles = parse_articles(records, long_study=long_study, rev_paper=rev_paper, sys_rev=sys_rev, clinical_trial=clin_trial, meta_analysis=meta, rand_ct=rct)
        
        # Convert parsed articles to DataFrame
        st.session_state.df = pd.DataFrame(articles)  # Store df in session state
        placeholder.empty()

# Display the search results if they exist in the session state
if st.session_state.df is not None and not st.session_state.df.empty:
    st.subheader("Search Results")
    num_results = len(st.session_state.df.index)
    st.write(st.session_state.df, f'Total Number of Results: {num_results}')
    csv = st.session_state.df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name='pubmed_search_results.csv',
        mime='text/csv',
    )

st.divider()

st.subheader("Clustering Analysis")

c10, c11, c12, c13, _, c19 = st.columns([5,5,3,3,8,3])

with c10:
    cluster_method = st.selectbox('Select Clustering Algorithm', ("K-Means", "LDA", "DBSCAN", "Hierarchical"), index=0)

with c11:
    dimred_method = st.selectbox('Select Dimensionality Reduction', ("PCA", "t-SNE", "UMAP"), index=0)

with c12:
    num_clusters = st.number_input("Number of Clusters", 2, 10, 5, 1)

with c13:
    n_keywords = st.number_input("Number of Keywords", 1, 20, 10, 1)

with c19:
    cluster_btn = st.button("Cluster", type='primary')

# Clustering Analysis section
if cluster_btn:
    if st.session_state.df is None or st.session_state.df.empty:
        st.write("No data to cluster!")
    else:
        cluster_sizes, cluster_keywords = run_cluster(
            st.session_state.df, n_clusters=num_clusters, n_key_words=n_keywords, method=cluster_method, dimred=dimred_method
        )
        
        if cluster_sizes.empty:
            st.write("No valid clusters were found with the current settings.")
        else:
            st.session_state.cluster_sizes = cluster_sizes
            st.session_state.cluster_keywords = cluster_keywords
            
            # Display the DataFrame again with cluster labels
            st.write("### Updated Search Results with Clusters", st.session_state.df)
            csv_clustered = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Download clustered results as CSV",
                data=csv_clustered,
                file_name='pubmed_clustered_results.csv',
                mime='text/csv',
            )
            c20, c21 = st.columns([2,5])
            with c20:
                st.write("Cluster Sizes:", st.session_state.cluster_sizes)
            with c21:
                st.write("Top Keywords per Cluster:", pd.DataFrame(st.session_state.cluster_keywords))