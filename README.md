# Hindi Media Analysis on Farmers' Protest

## Overview
This project analyzes **Hindi media coverage** of the **Farmers' Protest** using **topic modeling** and **natural language processing (NLP)** techniques. It aims to uncover dominant themes, sentiment trends, and media framing in news articles and reports.

## Project Goals
- Perform **topic modeling** on Hindi news articles related to the **Farmers' Protest**  
- Compare topics across different news sources to identify **biases and patterns**  
- Use **LLaMA** to enhance results from **BERTopic** for **better topic coherence**  
- Analyze **sentiment** and **key themes** in media coverage  

## Dataset
The dataset consists of Hindi news articles related to the **Farmers' Protest**, collected from multiple online sources. Preprocessing steps include:  
- **Tokenization & Stopword Removal** (specific to Hindi)  
- **Normalization** (handling different script variations)  
- **Translation (if needed)** to compare with English media coverage  

## Methodology

### 1. Data Preprocessing
- Used **Hindi NLP libraries** for tokenization and stopword removal  
- Cleaned data by removing irrelevant content, special characters, and duplicates  

### 2. Topic Modeling
- Applied **BERTopic** for **unsupervised topic detection**  
- Enhanced results using **LLaMA** to improve **coherence and interpretability**  
- Clustered articles based on dominant topics  

### 3. Sentiment Analysis
- Analyzed **sentiment polarity** (positive, neutral, negative) in different sources  
- Compared sentiment trends across different periods of the protest  

## Results & Insights
- Identified **key topics** (e.g., government policies, farmer demands, protests, arrests, media narratives)  
- Found **differences in topic emphasis** across various news outlets  
- Discovered **shifts in sentiment** over time based on major protest events  

## Technologies Used
- **Python** (NLP & ML processing)  
- **BERTopic** (Topic modeling)  
- **LLaMA** (Enhancing topic coherence)  
- **spaCy** & **IndicNLP** (Hindi text processing)  
- **Matplotlib & Seaborn** (Visualization)  

## Future Work
- Expand dataset with **more diverse media sources**  
- Implement **interactive dashboards** for visualizing trends  
- Apply **stance detection** to classify pro-farmer vs. anti-farmer narratives  

## Contributors
- **[Your Name]** (Lead Researcher & Developer)  

## License
This project is released under the **MIT License**.
