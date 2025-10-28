# MedScrape: AI-Assisted Literature Review Application

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medreview.streamlit.app/)

An AI-powered web application for automated literature search, clustering, and research gap identification using unsupervised learning and natural language processing.

**Developed by:** Department of Mathematical Sciences, United States Military Academy, West Point, NY

## 🌟 Features

### Core Functionality
- **Automated PubMed Scraping**: Query PubMed database with custom search terms and date ranges
- **Multiple Clustering Algorithms**: 
  - K-Means
  - DBSCAN
  - Hierarchical (Agglomerative)
  - Latent Dirichlet Allocation (LDA)
- **Advanced Embeddings**:
  - TF-IDF (fast, traditional)
  - Transformer-based (BERT) for semantic understanding
- **Dimensionality Reduction**:
  - PCA (Principal Component Analysis)
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)
  - UMAP (Uniform Manifold Approximation and Projection)

### Unique Features
- **Review vs. Research Comparison**: Automatically compare review papers to research papers
- **Gap Identification**: Use cosine similarity to identify underrepresented research areas
- **Interactive Visualizations**: Explore clusters with Plotly-powered interactive plots
- **Extractive Summarization**: Automatic TF-IDF-based summaries for each cluster
- **Export Capabilities**: Download results, cluster assignments, and analysis reports

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for full list of dependencies

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/lonespear/medscrape.git
cd medscrape

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app_enhanced.py
```

### Using the Online Version

Visit [https://medreview.streamlit.app/](https://medreview.streamlit.app/) to use the application without installation.

## 📖 Usage Guide

### Basic Literature Analysis

1. **Configure Search**:
   - Choose "Single Search" mode in the sidebar
   - Select "Scrape PubMed" as data source
   - Enter your email (required by NCBI)
   - Input your search query (e.g., `("genetics" OR "GWAS") AND "exercise"`)
   - Set date range and maximum results

2. **Collect Data**:
   - Navigate to "Data Collection" tab
   - Click "Start PubMed Scrape"
   - Review the collected articles and publication trends

3. **Run Clustering**:
   - Go to "Clustering" tab
   - Configure clustering parameters in sidebar:
     - Embedding method (TF-IDF or Transformer)
     - Clustering algorithm
     - Number of clusters (if applicable)
     - Visualization method
   - Click "Run Clustering"
   - Explore interactive visualizations and cluster summaries

4. **Export Results**:
   - Navigate to "Export" tab
   - Download clustered articles, summaries, or visualizations

### Advanced: Gap Analysis

1. **Configure Comparison**:
   - Select "Review vs Non-Review Comparison" mode
   - Configure PubMed search as above
   - The app will automatically scrape both review and research papers

2. **Identify Gaps**:
   - Navigate to "Gap Analysis" tab
   - Click "Identify Research Gaps"
   - Examine the coverage heatmap showing how well reviews cover research clusters
   - Review detailed gap reports for underrepresented areas

3. **Interpret Results**:
   - Red areas in heatmap = research gaps (low coverage by reviews)
   - Green areas = well-covered research domains
   - Export gap analysis for further investigation

### Upload Custom Data

Instead of scraping PubMed, you can upload your own CSV file:

**Required columns:**
- `Title`: Article title
- `Abstract`: Full abstract text
- `Authors`: Author list
- `Journal`: Journal name
- `Year`: Publication year

## 🔬 Methodology

This application implements the methods described in our paper:

**"Literature Reviews in the Age of Precision Nutrition"**  
*Beckman JR, Day J, Nelson R, Weeks Q, Dorta J, Doumbia M, Thomas DM*

### Clustering Pipeline

1. **Text Preprocessing**: Tokenization, stop word removal, lemmatization
2. **Feature Extraction**: 
   - TF-IDF vectorization OR
   - Transformer embeddings (sentence-transformers)
3. **Clustering**: Apply selected algorithm (K-Means, DBSCAN, Hierarchical, or LDA)
4. **Dimensionality Reduction**: Project to 2D for visualization
5. **Summarization**: Extract top terms using TF-IDF

### Gap Identification

1. **Separate Clustering**: Cluster review and research papers independently
2. **Aggregate by Cluster**: Combine all abstracts within each cluster
3. **Vectorization**: Create TF-IDF vectors for cluster-level documents
4. **Cosine Similarity**: Compute pairwise similarity between review and research clusters
5. **Threshold-based Detection**: Identify research clusters with low similarity to all review clusters

## 📊 Example Use Cases

### Precision Nutrition Research
```
Query: ("genetics" OR "GWAS" OR "genomic") AND ("nutrition" OR "diet" OR "metabolism")
```

### Exercise Physiology
```
Query: ("genes" OR "genetics") AND ("exercise" OR "physical activity" OR "training")
```

### Clinical Medicine
```
Query: ("diabetes" OR "obesity") AND ("treatment" OR "intervention") AND ("randomized controlled trial"[PT])
```

## 🛠️ Development

### Project Structure

```
medscrape/
├── app_enhanced.py          # Main Streamlit application
├── clustering_enhanced.py   # Clustering and comparison classes
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Key Classes

**`EnhancedClusterer`**
- Handles text preprocessing
- Creates TF-IDF or transformer embeddings
- Performs clustering with multiple algorithms
- Generates cluster summaries

**`ReviewComparator`**
- Compares review vs. research papers
- Computes cosine similarity matrices
- Identifies research gaps
- Creates visualization heatmaps

### Adding New Features

To add new clustering algorithms:

```python
def cluster_my_algorithm(self, features: np.ndarray, **kwargs) -> np.ndarray:
    """Your custom clustering method"""
    # Implement clustering
    return cluster_labels
```

To add new embedding methods:

```python
def create_my_embeddings(self, model_name: str) -> np.ndarray:
    """Your custom embedding method"""
    # Create embeddings
    return embeddings
```

## 📝 Citation

If you use this tool in your research, please cite:

```bibtex
@article{beckman2024literature,
  title={Literature Reviews in the Age of Precision Nutrition},
  author={Beckman, Jake R and Day, Jon and Nelson, Russell and Weeks, Quinten and Dorta, Joseph and Doumbia, Moussa and Thomas, Diana M},
  journal={TBD},
  year={2024},
  institution={United States Military Academy}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Jake R. Beckman** - United States Military Academy
- **Jon Day** - United States Military Academy  
- **Russell Nelson** - United States Military Academy
- **Quinten Weeks** - United States Military Academy
- **Joseph Dorta** - United States Military Academy
- **Moussa Doumbia** - United States Military Academy
- **Diana M. Thomas** - United States Military Academy

## 📧 Contact

- **Dr. Diana Thomas** - diana.thomas@westpoint.edu
- **Project Link**: https://github.com/lonespear/medscrape

## 🙏 Acknowledgments

- Nutrition for Precision Health (NPH) Initiative
- All of Us Research Program
- Anthropic Claude for AI assistance in development
- Streamlit for the excellent web framework

## 🔮 Future Enhancements

- [ ] Integration with additional databases (Web of Science, Scopus)
- [ ] Citation network analysis
- [ ] Author collaboration networks
- [ ] Temporal trend analysis
- [ ] API endpoints for programmatic access
- [ ] Multi-language support
- [ ] Advanced filtering by journal impact factor
- [ ] Automated email alerts for new publications

## 📚 Related Resources

- [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- [Nutrition for Precision Health](https://commonfund.nih.gov/nutritionforprecisionhealth)
- [All of Us Research Program](https://allofus.nih.gov/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)

---

**Made with ❤️ at the United States Military Academy**
