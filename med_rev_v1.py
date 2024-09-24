import streamlit as st
from streamlit_tags import st_tags
from Bio import Entrez

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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
        if not randomized_controlled_trial:
            for row in data:
                row.pop("RCT", None)
    Parse the PubMed articles to extract relevant information.
    
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

        # Flag whether the abstract mentions "risk of breast cancer"
        LCIS = "Lobular carcinoma in situ" in abstract.lower()
        
        # Flag whether the abstract mentions terms indicating a longitudinal study
        longitudinal_terms = ["longitudinal", "long-term follow up", "long term follow up", "follow-up", "follow up"]
        longitudinal_study = any(term in abstract.lower() for term in longitudinal_terms)

        # Medical Terms
        breast_cancer_terms = ["breast cancer", "mammary carcinoma", "invasive ductal carcinoma (IDC)", 
                               "invasive lobular carcinoma (ILC)", "ductal carcinoma in situ (DCIS)", 
                               "lobular carcinoma in situ (LCIS)", "triple-negative breast cancer",
                               "HER2-positive breast cancer", "BRCA1 mutations", "BRCA2 mutations", 
                               "metastatic breast cancer", "hormone receptor-positive breast cancer", 
                               "estrogen receptor-positive (ER-positive)", "progesterone receptor-positive (PR-positive)"]
        breast_cancer = any(term in abstract.lower() for term in breast_cancer_terms)

        # Append the extracted and flagged data to the list
        data.append({
            "PublicationYear": pub_year,
            "Title": title,
            "Authors": authors,
            "Abstract": abstract,
            "Review": 1 if is_review else 0,
            "SysRev": 1 if is_sys_rev else 0,
            "ClinicalTrial": 1 if is_clinical_trial else 0,
            "MetaAnalysis": 1 if is_meta_analysis else 0,
            "RCT": 1 if is_randomized_controlled_trial else 0,
            "LongitudinalStudy": 1 if longitudinal_study else 0,
            "LCIS": 1 if LCIS else 0,
            "BreastCancer": 1 if breast_cancer else 0,
            "PubMedURL": url,
            "JournalName": journal_name,
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

def run_cluster(df, n_clusters, n_key_words, method, dimred):
    # Ensure we are using the K-Means method with PCA dimensionality reduction for this test
    if method == "K-Means" and dimred == "PCA":
        text_data = df['Title'].fillna('') + ' ' + df['Abstract'].fillna('')
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        X = vectorizer.fit_transform(text_data)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        df['Cluster'] = kmeans.labels_
        
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

search_terms = [
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
    "Lobular carcinoma in situ",
    "Breast cancer", 
    "Mammary carcinoma", 
    "Invasive ductal carcinoma (IDC)", 
    "Invasive lobular carcinoma (ILC)", 
    "Ductal carcinoma in situ (DCIS)", 
    "Lobular carcinoma in situ (LCIS)", 
    "Triple-negative breast cancer", 
    "HER2-positive breast cancer", 
    "BRCA1 mutations", 
    "BRCA2 mutations", 
    "Metastatic breast cancer", 
    "Hormone receptor-positive breast cancer", 
    "Estrogen receptor-positive (ER-positive)", 
    "Progesterone receptor-positive (PR-positive)", 
    "Breast neoplasm", 
    "Breast tumor", 
    "Oncogene (related to breast cancer)", 
    "Breast cancer survival", 
    "Breast cancer risk factors", 
    "Breast cancer prevention", 
    "Breast cancer recurrence", 
    "Breast cancer metastasis", 
    "BRCA mutation", 
    "Genetic predisposition", 
    "Family history of breast cancer", 
    "Hormone replacement therapy (HRT)", 
    "upgrade", 
    "upstage", 
    "upstaging"
]

# Instantiate df
# Initialize df in session state if not already present
if 'df' not in st.session_state:
    st.session_state.df = None

st.set_page_config(layout="wide")

# Title
st.title("Systematic Review Tool for PubMed")

st.subheader("Scraping Tool")

included = st_tags(
    value=search_terms,
    label="#### Include words or phrases:",
    text="Press enter to add more",
    key='included'
)

excluded = st_tags(
    label="##### Exclude words or phrases:",
    text="Press enter to add more",
    maxtags=4,
    key='excluded'
)

query_inc = " OR ".join([f'"{term}"' for term in included])
query_ex = " AND ".join([f'NOT "{term}"' for term in excluded])
query = f'({query_inc}) {query_ex}'

st.write('##### Final Query')
st.write(query)

c3, c4, c5, _ , c9= st.columns([3, 3, 3, 8, 3])

with c3:
    max_results = st.number_input('Maximum Results', min_value=500, max_value=20000, value=2000, step=1)

with c4:
    long_study = st.checkbox("Longitudinal Study", value = True)
    rev_paper = st.checkbox("Review Paper", value = True)

with c5:
    sys_rev = st.checkbox("Systematic Review", value = True)
    clinical_trial = st.checkbox("Review Paper", value = True)

with c9:
    run_query = st.button("Run Search", type='primary')

# Perform the search when the button is clicked
if run_query:
    if included == []:
        st.write("Query is empty!")
        run_query = False
    else:
        st.write("Searching PubMed...")
        # Provide your email for the Entrez API
        Entrez.email = "your.email@example.com"  # Replace with your email
        records = search_pubmed(query, max_results=max_results)
        
        # Parse the search results
        articles = parse_articles(records, long_study=True, rev_paper=True, sys_rev=True, clinical_trial=True, meta_analysis=True, rand_ct=True)
        
        # Convert parsed articles to DataFrame
        st.session_state.df = pd.DataFrame(articles)  # Store df in session state

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
            c20, c21 = st.columns([5,5])
            with c20:
                st.write("Cluster Sizes:", st.session_state.cluster_sizes)
            with c21:
                st.write("Top Keywords per Cluster:", pd.DataFrame(st.session_state.cluster_keywords))


   
def cluster_and_filter_relevance(df, n_clusters=5, n_key_words=10):
    """
    Perform K-means clustering on the articles' abstracts and filter the most relevant clusters.
    
    :param df: DataFrame containing the articles data.
    :param n_clusters: Number of clusters to create.
    :param n_key_words: Number of top keywords to use for filtering relevant clusters.
    :return: Filtered DataFrame with relevant clusters, and a dictionary containing cluster keywords.
    """
    # Combine title and abstract into a single text field
    df['Title_Abstract'] = df['Title'] + ' ' + df['Abstract']
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Title_Abstract'].fillna(''))
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    df['Cluster'] = kmeans.labels_
    
    # Analyze the clusters to determine relevance
    # Initialize a dictionary to store the keywords for each cluster
    cluster_keywords = {}
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    # Iterate through each cluster and store the top n keywords
    for i in range(n_clusters):
        cluster_keywords[i] = [terms[ind] for ind in order_centroids[i, :n_key_words]]
    
    # Here you could filter clusters based on relevance, or simply drop the combined column
    df_filtered = df.copy()  # If you want to perform further filtering, modify df_filtered
    
    # Remove the Title_Abstract column before returning
    df_filtered.drop(columns=['Title_Abstract'], inplace=True)
    
    # Return both the filtered DataFrame and the cluster_keywords dictionary
    return df_filtered, cluster_keywords