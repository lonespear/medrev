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
from clustering_enhanced import EnhancedClusterer, ReviewComparator, create_interactive_plot
import plotly.express as px
import plotly.graph_objects as go

# NCBI API key — raises rate limit from 3 req/s to 10 req/s
NCBI_API_KEY = "5a3c321d881ae6f0902a1ba96538a24e4808"

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


def count_pubmed_results(query: str, email: str, start_year: int = None, end_year: int = None) -> int:
    """Return the number of PubMed records matching query without fetching them."""
    Entrez.email = email
    Entrez.api_key = NCBI_API_KEY
    full_query = query
    if start_year and end_year:
        full_query += f' AND {start_year}:{end_year}[pdat]'
    try:
        handle = Entrez.esearch(db="pubmed", term=full_query, retmax=0)
        record = Entrez.read(handle)
        handle.close()
        return int(record["Count"])
    except Exception:
        return -1


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
    Entrez.api_key = NCBI_API_KEY

    # Add publication type filter
    if publication_type == "review":
        query += ' AND "review"[Publication Type]'
    elif publication_type == "non-review":
        query += ' NOT "review"[Publication Type]'
    
    # Add date range if specified
    if start_year and end_year:
        query += f' AND {start_year}:{end_year}[pdat]'
    
    # Search — usehistory stores results server-side so there is no ID-list cap
    try:
        handle = Entrez.esearch(db="pubmed", term=query, usehistory="y")
        record = Entrez.read(handle)
        total_count = int(record["Count"])
        webenv     = record["WebEnv"]
        query_key  = record["QueryKey"]
        handle.close()

        fetch_count = min(total_count, max_results)

        if fetch_count == 0:
            return pd.DataFrame()

        # Fetch details in batches using server-side history (no ID list needed)
        batch_size = 500
        all_records = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for retstart in range(0, fetch_count, batch_size):
            batch_end = min(retstart + batch_size, fetch_count)

            status_text.text(
                f"Fetching records {retstart + 1:,}–{batch_end:,} of {fetch_count:,}"
                f"  (total matches on PubMed: {total_count:,})"
            )
            progress_bar.progress(batch_end / fetch_count)

            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    rettype="medline", retmode="xml",
                    retstart=retstart, retmax=batch_size,
                    webenv=webenv, query_key=query_key,
                )
                records = Entrez.read(handle)
                handle.close()
            except Exception as batch_err:
                st.warning(f"Batch {retstart + 1}–{batch_end} failed ({batch_err}). Skipping and continuing...")
                time.sleep(2)
                continue

            for record in records['PubmedArticle']:
                try:
                    article = record['MedlineCitation']['Article']

                    authors = []
                    if 'AuthorList' in article:
                        for author in article['AuthorList']:
                            if 'LastName' in author and 'Initials' in author:
                                authors.append(f"{author['LastName']} {author['Initials']}")

                    abstract = ''
                    if 'Abstract' in article:
                        abstract_parts = article['Abstract'].get('AbstractText', [])
                        if abstract_parts:
                            abstract = ' '.join([str(part) for part in abstract_parts])

                    year = None
                    if 'ArticleDate' in article and article['ArticleDate']:
                        year = int(article['ArticleDate'][0]['Year'])
                    elif 'Journal' in article and 'JournalIssue' in article['Journal']:
                        pub_date = article['Journal']['JournalIssue'].get('PubDate', {})
                        if 'Year' in pub_date:
                            year = int(pub_date['Year'])

                    article_type_list = article.get('PublicationTypeList', [])
                    is_review = any(pub_type.lower() == "review" for pub_type in article_type_list)
                    is_sys_rev = any(pub_type.lower() == "systematic review" for pub_type in article_type_list)
                    is_clinical_trial = any(pub_type.lower() == "clinical trial" for pub_type in article_type_list)
                    is_meta_analysis = any(pub_type.lower() == "meta-analysis" for pub_type in article_type_list)
                    is_rct = any(pub_type.lower() == "randomized controlled trial" for pub_type in article_type_list)

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
                except Exception:
                    continue

            time.sleep(0.11)  # ~9 req/s — safely under the 10 req/s API-key limit

        progress_bar.empty()
        status_text.empty()

        if all_records:
            return pd.DataFrame(all_records)
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error scraping PubMed: {str(e)}")
        # Return whatever was collected before the error
        if all_records:
            st.warning(f"Returning {len(all_records):,} records collected before the error.")
            return pd.DataFrame(all_records)
        return pd.DataFrame()


def export_results(df: pd.DataFrame, filename: str = "results.csv") -> bytes:
    """Export DataFrame to CSV"""
    return df.to_csv(index=False).encode('utf-8')


def build_query_from_groups(query_groups, field="[Title/Abstract]"):
    """
    Build PubMed query string from query groups.

    Each term is tagged with the chosen field qualifier (e.g. [Title/Abstract])
    so the search is scoped properly instead of hitting all 30M+ records in the
    full-text index.  Terms within a group are OR'd; groups are joined by the
    selected inter-group operator (AND / OR / NOT).  Every group is always
    wrapped in parentheses so nesting is unambiguous.

    Args:
        query_groups: List of dicts with 'terms' (list[str]) and 'operator' keys
        field: PubMed field qualifier string, e.g. "[Title/Abstract]"

    Returns:
        Query string ready to send to Entrez
    """
    if not query_groups or all(not group['terms'] for group in query_groups):
        return ""

    query_parts = []
    for i, group in enumerate(query_groups):
        if not group['terms']:
            continue

        # Tag each term with the field qualifier
        tagged = [f'"{term}"{field}' for term in group['terms']]

        # Always wrap the group in parentheses for unambiguous nesting
        group_query = f"({' OR '.join(tagged)})"

        if i == 0:
            query_parts.append(group_query)
        else:
            operator = group['operator']
            query_parts.append(f"{operator} {group_query}")

    return " ".join(query_parts)


def create_intelligent_publication_trends(df: pd.DataFrame):
    """
    Create intelligent, adaptive publication trends visualization based on data characteristics.

    Args:
        df: DataFrame with 'Year' column
    """
    if df is None or df.empty or 'Year' not in df.columns:
        st.warning("No data available for publication trends.")
        return

    # Remove NaN years and get statistics
    df_clean = df[df['Year'].notna()].copy()
    if df_clean.empty:
        st.warning("No valid year data available.")
        return

    year_counts = df_clean['Year'].value_counts().sort_index()
    unique_years = len(year_counts)
    year_range = int(df_clean['Year'].max() - df_clean['Year'].min())
    total_papers = len(df_clean)

    st.subheader("📊 Publication Trends")

    # Adaptive display based on year range
    if unique_years == 1:
        # Single year: Show summary card instead of plot
        single_year = int(df_clean['Year'].iloc[0])

        col1, col2, col3 = st.columns(3)
        col1.metric("Publication Year", single_year)
        col2.metric("Total Articles", total_papers)
        col3.metric("Year Range", "Single Year")

        st.info("""
        ℹ️ **Single-year dataset detected.** Timeline visualization is not helpful with data from only one year.
        Consider expanding your year range in the search parameters for trend analysis.
        """)

        # Show article type breakdown if available
        article_type_cols = ['Review', 'SystematicReview', 'ClinicalTrial',
                            'MetaAnalysis', 'RCT', 'LongitudinalStudy']
        if all(col in df_clean.columns for col in article_type_cols):
            st.markdown("**Article Type Distribution:**")
            type_data = {
                'Type': ['Reviews', 'Systematic Reviews', 'Clinical Trials',
                        'Meta-Analyses', 'RCTs', 'Longitudinal'],
                'Count': [df_clean[col].sum() for col in article_type_cols]
            }
            type_df = pd.DataFrame(type_data)
            type_df = type_df[type_df['Count'] > 0]  # Only show types with data

            fig = px.bar(
                type_df,
                x='Type',
                y='Count',
                title=f'Article Types in {single_year}',
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

    elif unique_years == 2:
        # Two years: Show year-over-year comparison
        years_list = sorted(year_counts.index)
        year1, year2 = int(years_list[0]), int(years_list[1])
        count1, count2 = year_counts[year1], year_counts[year2]

        change = count2 - count1
        pct_change = ((count2 - count1) / count1 * 100) if count1 > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{year1}", f"{count1} articles")
        col2.metric(f"{year2}", f"{count2} articles")
        col3.metric("Change", f"{change:+d} articles")
        col4.metric("Growth Rate", f"{pct_change:+.1f}%")

        # Simple comparison bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=[str(year1), str(year2)],
                y=[count1, count2],
                text=[count1, count2],
                textposition='auto',
                marker_color=['#636EFA', '#EF553B']
            )
        ])
        fig.update_layout(
            title='Year-over-Year Comparison',
            xaxis_title='Year',
            yaxis_title='Number of Publications',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        ℹ️ **Two-year dataset.** Consider expanding your year range to 3+ years for better trend analysis.
        """)

    else:
        # 3+ years: Full interactive timeline with trend analysis

        # Calculate trend metrics
        avg_per_year = total_papers / unique_years
        min_year, max_year = int(df_clean['Year'].min()), int(df_clean['Year'].max())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Year Range", f"{min_year} - {max_year}")
        col2.metric("Total Papers", total_papers)
        col3.metric("Years Covered", unique_years)
        col4.metric("Avg per Year", f"{avg_per_year:.1f}")

        # Create interactive Plotly timeline
        timeline_df = pd.DataFrame({
            'Year': year_counts.index,
            'Publications': year_counts.values
        })

        # Add trend line
        z = np.polyfit(timeline_df['Year'], timeline_df['Publications'], 1)
        p = np.poly1d(z)
        timeline_df['Trend'] = p(timeline_df['Year'])

        fig = go.Figure()

        # Bar chart
        fig.add_trace(go.Bar(
            x=timeline_df['Year'],
            y=timeline_df['Publications'],
            name='Publications',
            marker_color='#636EFA',
            hovertemplate='<b>Year %{x}</b><br>Publications: %{y}<extra></extra>'
        ))

        # Trend line
        fig.add_trace(go.Scatter(
            x=timeline_df['Year'],
            y=timeline_df['Trend'],
            name='Trend',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Trend</b><br>Year %{x}<br>Expected: %{y:.1f}<extra></extra>'
        ))

        fig.update_layout(
            title='Publication Timeline with Trend Analysis',
            xaxis_title='Year',
            yaxis_title='Number of Publications',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Growth analysis
        if unique_years >= 5:
            # Calculate year-over-year growth rates
            growth_rates = []
            for i in range(1, len(timeline_df)):
                prev_count = timeline_df.iloc[i-1]['Publications']
                curr_count = timeline_df.iloc[i]['Publications']
                if prev_count > 0:
                    growth_rate = ((curr_count - prev_count) / prev_count * 100)
                    growth_rates.append(growth_rate)

            if growth_rates:
                avg_growth = np.mean(growth_rates)

                st.markdown("### 📈 Growth Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Annual Growth", f"{avg_growth:.1f}%")
                col2.metric("Peak Year", f"{timeline_df['Publications'].idxmax()}")
                col3.metric("Peak Publications", f"{timeline_df['Publications'].max()}")

                # Trend interpretation
                if avg_growth > 10:
                    st.success("📈 Strong upward trend - Growing research interest in this area")
                elif avg_growth > 0:
                    st.info("➡️ Moderate growth - Steady research activity")
                elif avg_growth > -10:
                    st.warning("📉 Slight decline - Research interest may be stabilizing")
                else:
                    st.error("📉 Significant decline - Decreasing research activity")


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

        st.markdown("### 📋 Analysis Mode")
        analysis_mode = st.radio(
            "Select analysis type",
            ["Single Search", "Review vs Non-Review Comparison"],
            help="Single Search: Cluster all papers together. Comparison: Identify gaps between review and research papers."
        )

        if analysis_mode == "Single Search":
            st.info("💡 Performs clustering analysis on all collected papers to identify research themes.")
        else:
            st.info("💡 Compares review papers against research papers to identify under-reviewed topics (Gap Analysis).")

        st.markdown("---")

        # Data source
        st.markdown("### 📁 Data Source")
        data_source = st.radio(
            "Choose data input method",
            ["Scrape PubMed", "Upload CSV"],
            help="Scrape PubMed directly or upload your own CSV file"
        )
        
        if data_source == "Scrape PubMed":
            st.subheader("🔍 PubMed Search")
            email = st.text_input(
                "Email (required by NCBI)",
                value="user@example.com",
                help="NCBI requires an email address for API access. This is used for tracking and rate limiting."
            )

            st.markdown("### 📅 Search Parameters")
            col1, col2 = st.columns(2)
            with col1:
                start_year = st.number_input(
                    "Start Year",
                    min_value=1900,
                    max_value=2025,
                    value=2000,
                    help="Filter publications from this year onwards"
                )
            with col2:
                end_year = st.number_input(
                    "End Year",
                    min_value=1900,
                    max_value=2025,
                    value=2025,
                    help="Filter publications up to this year"
                )

            year_range = end_year - start_year
            if year_range == 0:
                st.warning("⚠️ Single-year search: trends will show limited data.")
            elif year_range == 1:
                st.warning("⚠️ Two-year search: recommend 3+ years for trend analysis.")
            elif year_range < 5:
                st.info("ℹ️ Short time span. Trend analysis will be basic.")

            st.markdown("### 🔎 Query Builder")
            with st.expander("ℹ️ How to Build Queries", expanded=False):
                st.markdown("""
                **Query Building Logic:**
                - **Within a group**: Keywords are combined with OR (any keyword matches)
                - **Between groups**: Groups are combined with AND/OR/NOT operators

                **Examples:**
                - Group 1: `nutrition, diet` → matches papers with "nutrition" OR "diet"
                - Group 2: `genetics, genomics` → matches papers with "genetics" OR "genomics"
                - Combined with AND: Papers must mention (nutrition OR diet) AND (genetics OR genomics)

                **Tips:**
                - Use multiple groups to narrow down results
                - Use NOT operator to exclude unwanted topics
                - Add more specific terms if you get too many results
                """)

            st.markdown("**Build your search query:**")

            # Field qualifier selector — critical for avoiding millions of false matches
            field_qualifier = st.selectbox(
                "Search field",
                ["[Title/Abstract]", "[MeSH Terms]", "[All Fields]"],
                index=0,
                help=(
                    "[Title/Abstract] (recommended): matches terms in title or abstract only. "
                    "[MeSH Terms]: controlled vocabulary, most precise. "
                    "[All Fields]: searches everything including affiliations — often returns millions of results."
                )
            )

            # Display and manage query groups
            for idx, group in enumerate(st.session_state.query_groups):
                if idx > 0:
                    op_key = f'group_{idx}_operator'
                    group['operator'] = st.selectbox(
                        f"Group {idx + 1} joins with:",
                        options=["AND", "OR", "NOT"],
                        index=["AND", "OR", "NOT"].index(group.get('operator', 'AND')),
                        key=op_key
                    )
                else:
                    st.markdown(f"**Group {idx + 1}**")

                terms_raw = st.text_input(
                    f"Keywords (comma-separated)",
                    value=", ".join(group.get('terms', [])),
                    key=f'group_{idx}_terms',
                    placeholder="e.g. artificial intelligence, machine learning",
                    help="Terms within a group are combined with OR."
                )
                group['terms'] = [t.strip() for t in terms_raw.split(',') if t.strip()]

                if len(st.session_state.query_groups) > 1:
                    if st.button(f"✕ Remove Group {idx + 1}", key=f'remove_group_{idx}'):
                        st.session_state.query_groups.pop(idx)
                        st.rerun()

                if idx < len(st.session_state.query_groups) - 1:
                    st.markdown("---")

            if st.button("➕ Add Group"):
                st.session_state.query_groups.append({'terms': [], 'operator': 'AND'})
                st.rerun()

            # Build and display final query — updates live on every keystroke
            query = build_query_from_groups(st.session_state.query_groups, field=field_qualifier)
            st.markdown("**Final Query:**")
            st.code(query if query else "(empty — add keywords above)")

            # Count check before committing to a full scrape
            col_cnt, col_msg = st.columns([1, 2])
            with col_cnt:
                check_count = st.button("🔢 Check Count", help="Query PubMed for result count without fetching records")
            with col_msg:
                if check_count:
                    if not query:
                        st.warning("Build a query first.")
                    else:
                        with st.spinner("Counting..."):
                            n = count_pubmed_results(query, email, start_year, end_year)
                        if n < 0:
                            st.error("Could not reach PubMed.")
                        elif n > 100_000:
                            st.error(f"{n:,} results — far too broad. Refine your query.")
                        elif n > 20_000:
                            st.warning(f"{n:,} results — large dataset, scraping will take time.")
                        elif n == 0:
                            st.warning("0 results — check your terms.")
                        else:
                            st.success(f"{n:,} results — looks good!")

            max_results = st.number_input(
                "Max Results",
                min_value=100,
                max_value=200000,
                value=10000,
                help=(
                    "Maximum papers to retrieve. PubMed has no hard cap — "
                    "the API key allows 10 req/s so 30,000 takes ~6 min, "
                    "100,000 takes ~20 min. Set to 0 to fetch ALL matches."
                )
            )

            if analysis_mode == "Review vs Non-Review Comparison":
                st.info("Will scrape both review and non-review papers")
        else:
            st.subheader("Upload Data")
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            st.markdown("""
            **Required columns:** Title, Abstract, Authors, Journal, Year
            """)
        
        st.markdown("---")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Data Collection", "🔍 Clustering", "🎯 Gap Analysis", "📊 Export"])
    
    # Tab 1: Data Collection
    with tab1:
        st.header("📥 Data Collection")
        st.markdown("Collect literature data from PubMed or upload your own CSV file.")
        
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
            st.markdown("---")
            st.subheader("📊 Data Preview & Statistics")

            df = st.session_state.scraped_data

            # Metrics with quick export
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Articles", len(df))
            col2.metric("Year Range", f"{df['Year'].min():.0f} - {df['Year'].max():.0f}")
            col3.metric("Unique Journals", df['Journal'].nunique())

            # Quick export button
            with col4:
                st.markdown("**Quick Export:**")
                csv_data = export_results(df, "raw_data.csv")
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"raw_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Download the raw scraped data before clustering"
                )

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

            # Intelligent publication trends
            create_intelligent_publication_trends(df)
            
        if st.session_state.review_data is not None:
            st.subheader("Review Papers Preview")
            st.dataframe(st.session_state.review_data.head(50), use_container_width=True)
    
    # Tab 2: Clustering
    with tab2:
        st.header("Clustering Analysis")

        if st.session_state.scraped_data is not None:
            # Clustering Settings
            with st.expander("⚙️ Clustering Settings", expanded=True):
                st.markdown("### Configure Clustering Parameters")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Embedding Method")
                    st.markdown("""
                    Embedding methods convert text into numerical vectors for clustering.
                    """)
                    embedding_type = st.selectbox(
                        "Select Embedding Method",
                        ["TF-IDF", "Doc2Vec", "FastText"],
                        help="How to convert abstracts into numerical representations"
                    )

                    # Detailed help for embedding methods
                    if embedding_type == "TF-IDF":
                        st.info("""
                        **TF-IDF (Term Frequency-Inverse Document Frequency)**
                        - ⚡ Fastest option (recommended for large datasets)
                        - 📊 Best for keyword-based clustering
                        - 🎯 Works well when documents differ by specific terms
                        - ⚠️ Doesn't capture semantic meaning or word order
                        """)
                    elif embedding_type == "Doc2Vec":
                        st.info("""
                        **Doc2Vec (Document Vectors)**
                        - 🧠 Captures semantic meaning and context
                        - 🐢 Moderate speed (requires training)
                        - 🎯 Good for finding conceptually similar papers
                        - 💡 Uses neural network-based embeddings
                        """)
                    else:  # FastText
                        st.info("""
                        **FastText (Subword Embeddings)**
                        - ⚡ Lightweight and relatively fast
                        - 🔤 Handles rare words and typos well
                        - 🎯 Good balance between speed and semantic understanding
                        - 💡 Uses character-level information
                        """)

                    st.markdown("---")

                    st.markdown("#### Visualization Method")
                    st.markdown("""
                    Dimensionality reduction projects high-dimensional embeddings into 2D for visualization.
                    """)
                    reduction_method = st.selectbox(
                        "Select Visualization Method",
                        ["PCA", "t-SNE", "UMAP"],
                        help="How to reduce dimensions for 2D plotting"
                    )

                    # Detailed help for visualization methods
                    if reduction_method == "PCA":
                        st.info("""
                        **PCA (Principal Component Analysis)**
                        - ⚡ Very fast, deterministic results
                        - 📐 Preserves global structure well
                        - ⚠️ Linear method, may not capture complex patterns
                        - 🎯 Best for: Initial exploration, large datasets
                        """)
                    elif reduction_method == "t-SNE":
                        st.info("""
                        **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
                        - 🎨 Excellent at revealing cluster structure
                        - 🐢 Slower, non-deterministic (different runs vary)
                        - 🎯 Preserves local structure (nearby points)
                        - 💡 Best for: Final visualizations, smaller datasets
                        """)
                    else:  # UMAP
                        st.info("""
                        **UMAP (Uniform Manifold Approximation)**
                        - ⚡ Faster than t-SNE, more consistent
                        - 🎨 Preserves both local and global structure
                        - 🎯 Good balance of speed and quality
                        - 💡 Best for: Large datasets, general purpose
                        """)

                with col2:
                    st.markdown("#### Clustering Algorithm")
                    st.markdown("""
                    Clustering algorithms group similar documents together.
                    """)
                    clustering_method = st.selectbox(
                        "Select Clustering Algorithm",
                        ["K-Means", "DBSCAN", "Hierarchical", "LDA"],
                        help="Algorithm to identify document clusters"
                    )

                    # Detailed help for clustering algorithms
                    if clustering_method == "K-Means":
                        st.info("""
                        **K-Means Clustering**
                        - ⚡ Fast and scalable
                        - 🎯 Creates spherical clusters of similar size
                        - ⚙️ Requires specifying number of clusters
                        - 💡 Best for: Well-separated topics, known cluster count
                        """)
                        n_clusters = st.slider(
                            "Number of Clusters",
                            min_value=3,
                            max_value=20,
                            value=8,
                            help="How many topic clusters to identify. 5-10 works well for most literature reviews."
                        )
                    elif clustering_method == "DBSCAN":
                        st.info("""
                        **DBSCAN (Density-Based Clustering)**
                        - 🔍 Automatically finds number of clusters
                        - 🎯 Can identify noise/outliers
                        - ⚙️ Sensitive to parameter settings (eps, min_samples)
                        - 💡 Best for: Irregular cluster shapes, unknown cluster count
                        - ⚠️ May mark many papers as noise with default settings
                        """)
                        st.markdown("**Advanced DBSCAN Parameters:**")
                        st.caption("eps=0.5, min_samples=5 (hardcoded for stability)")
                    elif clustering_method == "Hierarchical":
                        st.info("""
                        **Hierarchical Clustering**
                        - 🌳 Creates tree-like cluster hierarchy
                        - 🎯 Good for exploring relationships at multiple levels
                        - ⚙️ Requires specifying number of clusters
                        - 💡 Best for: Taxonomic analysis, nested topics
                        """)
                        n_clusters = st.slider(
                            "Number of Clusters",
                            min_value=3,
                            max_value=20,
                            value=8,
                            help="Top-level clusters to extract from the hierarchy."
                        )
                    else:  # LDA
                        st.info("""
                        **LDA (Latent Dirichlet Allocation)**
                        - 📝 Probabilistic topic modeling
                        - 🎯 Documents can belong to multiple topics
                        - ⚙️ Generates interpretable word distributions per topic
                        - 💡 Best for: Topic modeling, mixed-topic documents
                        - ⏱️ Slower than other methods
                        """)
                        n_clusters = st.slider(
                            "Number of Topics",
                            min_value=3,
                            max_value=20,
                            value=8,
                            help="How many latent topics to discover in the literature."
                        )

                st.markdown("---")
                st.markdown("### 💡 Quick Recommendations")
                st.markdown("""
                - **New users**: Start with TF-IDF + K-Means + PCA (fastest, most reliable)
                - **Semantic analysis**: Use Doc2Vec + K-Means + UMAP (better topic coherence)
                - **Large datasets (>5000 papers)**: TF-IDF + K-Means + PCA or UMAP
                - **Unknown topic count**: DBSCAN (but review results carefully)
                - **Topic modeling**: LDA + TF-IDF (most interpretable topics)
                """)

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
                    label = summary.get('label', f'Cluster {cluster_id}')
                    with st.expander(f"Cluster {cluster_id} — {label}  ({summary['size']:,} papers)"):
                        headline = summary.get('headline', '')
                        if headline:
                            st.markdown(f"> {headline}")
                        st.markdown(f"**Key terms:** {', '.join(summary['top_terms'][:10])}")
                        titles = summary.get('sample_titles', [])
                        urls = summary.get('sample_urls', [])
                        if titles:
                            st.markdown("**Most representative papers:**")
                            for i, title in enumerate(titles):
                                url = urls[i] if i < len(urls) else None
                                if url:
                                    st.markdown(f"- [{title}]({url})")
                                else:
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
        st.header("🎯 Gap Analysis: Review vs Research Papers")
        st.markdown("""
        Identify research topics that are under-reviewed in the literature. This analysis compares
        clusters of research papers against clusters of review papers to find gaps in coverage.
        """)

        if analysis_mode == "Single Search":
            st.info("💡 Gap analysis is only available in 'Review vs Non-Review Comparison' mode. Switch modes in the sidebar to enable this feature.")
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

                st.markdown("---")
                st.subheader("📊 Coverage Heatmap")

                with st.expander("ℹ️ How to Interpret the Heatmap", expanded=False):
                    st.markdown("""
                    **Understanding the Heatmap:**
                    - **X-axis**: Review paper clusters
                    - **Y-axis**: Research paper clusters
                    - **Colors**: Similarity scores (0 = no overlap, 1 = identical)
                        - 🟢 **Green (>0.5)**: Research cluster is well-covered by reviews
                        - 🟡 **Yellow (0.3-0.5)**: Moderate coverage
                        - 🔴 **Red (<0.3)**: Research gap - poorly covered by existing reviews

                    **What This Means:**
                    - Red rows indicate research topics that lack comprehensive review coverage
                    - These are potential targets for new review papers
                    - Green rows indicate research areas already well-synthesized
                    """)

                st.markdown("""
                This heatmap shows the similarity between research and review paper clusters.
                **Red areas** indicate research gaps where existing reviews provide limited coverage.
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
        st.header("📊 Export Results")
        st.markdown("""
        Download your analysis results in CSV format for further analysis, reporting, or publication.
        All exports include timestamps for version tracking.
        """)
        
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
        <p>📧 Contact: jonathan.day@westpoint.edu | 🔗 <a href='https://github.com/lonespear/medrev'>GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
