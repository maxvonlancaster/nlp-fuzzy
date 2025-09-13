import streamlit as st
import networkx as nx
import itertools
import matplotlib.pyplot as plt


# st.write("""
# # My first app
# Hello *world!*
# """)



def preprocess_text(text):
    words = text.lower().split()
    return words

def build_co_occurrence_graph(words, window_size=2):
    G = nx.Graph()
    pairs = list(itertools.combinations(words, window_size))
    for pair in pairs:
        if G.has_edge(pair[0], pair[1]):
            G[pair[0]][pair[1]]['weight'] += 1
        else:
            G.add_edge(pair[0], pair[1], weight=1)
    return G

def apply_pagerank(G):
    pagerank_scores = nx.pagerank(G, weight='weight')
    return pagerank_scores

def generate_graph(G, pagerank_scores):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    
    nx.draw_networkx_nodes(G, pos, node_size=[v * 10000 for v in pagerank_scores.values()], node_color='skyblue')
    
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("Co-occurrence Graph with PageRank Scores")
    return plt



st.set_page_config(layout="wide")

st.title("Text Summarizer Comparisons")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Text")
    text_input = st.text_area("Enter your text here:", height=300)

with col2:
    st.subheader("Summary")
    if text_input:
        words = preprocess_text(text_input)
        G = build_co_occurrence_graph(words)
        pagerank_scores = apply_pagerank(G)

        plt = generate_graph(G, pagerank_scores)
        st.pyplot(plt)
    else:
        st.info("Please enter text on the left to see the summary here.")