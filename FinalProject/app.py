from flask import Flask, request, render_template
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import pandas as pd

app = Flask(__name__)

# Core Functions
def train_word2vec_model(words, vector_size=100):
    """Trains a Word2Vec model on a list of words."""
    model = Word2Vec([words], min_count=1, vector_size=vector_size)
    return model

def filter_words(words, model):
    """Filters words to ensure they are present in the model's vocabulary."""
    return [word for word in words if word in model.wv.key_to_index]

def extract_word_pairs(words):
    """Extracts all pairs of words."""
    pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            pairs.append((words[i], words[j]))
    return pairs

def calculate_cosine_similarity(word_vectors, word_pairs):
    """Calculates cosine similarity between word pairs."""
    similarities = []
    for word1, word2 in word_pairs:
        vector1 = word_vectors[word1]
        vector2 = word_vectors[word2]
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        similarities.append((similarity, word1, word2))
    return sorted(similarities, reverse=True)  # Sort by similarity in descending order

def form_8_unique_pairs(similarities):
    """Forms the top 8 unique word pairs based on similarity."""
    used_words = set()
    pairs = []
    for similarity, word1, word2 in similarities:
        if word1 not in used_words and word2 not in used_words:
            pairs.append((word1, word2))
            used_words.add(word1)
            used_words.add(word2)
            if len(pairs) == 8:
                break
    return pairs

def group_word_pairs(pairs, similarity_matrix, words):
    """Groups word pairs into groups based on similarity matrix."""
    groups = []
    available_pairs = list(range(len(pairs)))

    while len(available_pairs) >= 2:
        max_similarity = -1
        best_pair_indices = None

        for i in available_pairs:
            for j in available_pairs:
                if i != j:
                    pair1, pair2 = pairs[i], pairs[j]
                    avg_similarity = np.mean([
                        similarity_matrix[words.index(w1), words.index(w2)]
                        for w1 in pair1 for w2 in pair2
                    ])

                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        best_pair_indices = (i, j)

        if best_pair_indices:
            i, j = best_pair_indices
            groups.append(pairs[i] + pairs[j])
            available_pairs.remove(i)
            available_pairs.remove(j)

    return groups

def compute_success_rate(original_groups, generated_groups):
    """Computes the success rate between original and generated groups."""
    matched_groups = 0

    for generated_group in generated_groups:
        # Check if the generated group exactly matches any original group
        for original_group in original_groups:
            if set(generated_group) == set(original_group):
                matched_groups += 1
                break  # Stop checking once a match is found

    total_groups = len(original_groups)
    success_rate = (matched_groups / total_groups) * 100 if total_groups else 0
    return success_rate

def calculate_success_rate(original_groups, generated_groups):
    """
    Calculate the success rate of grouping words based on the overlap between original and generated groups.

    :param original_groups: List of original groups (list of lists)
    :param generated_groups: List of generated groups (list of lists)
    :return: Success rate as a percentage
    """
    total_words = sum(len(group) for group in original_groups)
    correctly_grouped = 0

    # Check matches between original and generated groups
    for original_group in original_groups:
        max_overlap = 0
        for generated_group in generated_groups:
            # Count overlapping words
            overlap = len(set(original_group) & set(generated_group))
            max_overlap = max(max_overlap, overlap)
        correctly_grouped += max_overlap

    # Calculate success rate
    success_rate = (correctly_grouped / total_words) * 100
    return success_rate

def generate_pca_plot(model, filtered_words, groups):
    """Generates a PCA plot for word embeddings."""
    word_vectors = [model.wv[word] for word in filtered_words]
    
    # Perform PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)

    # Create a DataFrame for Plotly
    df = pd.DataFrame(reduced_vectors, columns=["PCA Component 1", "PCA Component 2"])
    df['Word'] = filtered_words
    
    # Assign colors to words based on their group
    word_to_group = {}
    group_colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta']  # Adjust as necessary
    for i, group in enumerate(groups):
        for word in group:
            word_to_group[word] = group_colors[i % len(group_colors)]
    df['Color'] = df['Word'].apply(lambda word: word_to_group.get(word, 'black'))

    # Create interactive PCA plot using Plotly
    fig = px.scatter(df, x="PCA Component 1", y="PCA Component 2", text="Word", color="Color",
                     title="Word Embedding Visualization (PCA)", 
                     labels={"PCA Component 1": "PCA Component 1", "PCA Component 2": "PCA Component 2"})
    fig.update_traces(marker=dict(size=12),
                      textposition='top center', 
                      showlegend=False)

    # Convert to HTML for embedding in Flask
    return fig.to_html(full_html=False)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        words = request.form["words"].split(",")
        original_groups_input = request.form["original_groups"]

        # Convert original groups input into list of lists
        original_groups = [group.split(",") for group in original_groups_input.split(";")]

        # Train Word2Vec model
        model = train_word2vec_model(words)

        # Filter words to ensure they are present in the model's vocabulary
        filtered_words = filter_words(words, model)

        if not filtered_words:
            return render_template("index.html", error="No valid words found in the model's vocabulary.")

        # Extract word pairs
        word_pairs = extract_word_pairs(filtered_words)

        # Create a dictionary of word vectors for easy access
        word_vectors = {word: model.wv[word] for word in filtered_words}

        # Calculate cosine similarities
        similarities = calculate_cosine_similarity(word_vectors, word_pairs)

        if not similarities:
            return render_template("index.html", error="No similarities calculated.")

        # Form top 8 unique pairs
        top_8_pairs = form_8_unique_pairs(similarities)

        # Prepare similarity matrix for grouping
        similarity_matrix = np.zeros((len(filtered_words), len(filtered_words)))

        for i, word1 in enumerate(filtered_words):
            for j, word2 in enumerate(filtered_words):
                similarity_matrix[i, j] = np.dot(
                    word_vectors[word1], word_vectors[word2]
                ) / (np.linalg.norm(word_vectors[word1]) * np.linalg.norm(word_vectors[word2]))

        # Group word pairs into groups
        grouped_pairs = group_word_pairs(top_8_pairs, similarity_matrix, filtered_words)

        # Compute success rate based on at least two matching words
        success_rate = calculate_success_rate(original_groups, grouped_pairs)

        # Generate interactive PCA plot for word embeddings
        pca_plot = generate_pca_plot(model, filtered_words, grouped_pairs)

        return render_template("index.html", words=words, original_groups=original_groups, 
                               generated_groups=grouped_pairs, success_rate=success_rate, pca_plot=pca_plot)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)