from collections import defaultdict
import pprint
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Input data
sentences = [
    "The_DET cat_NOUN sleeps_VERB",
    "A_DET dog_NOUN barks_VERB",
    "The_DET dog_NOUN sleeps_VERB",
    "My_DET dog_NOUN runs_VERB fast_ADV",
    "A_DET cat_NOUN meows_VERB loudly_ADV",
    "Your_DET cat_NOUN runs_VERB",
    "The_DET bird_NOUN sings_VERB sweetly_ADV",
    "A_DET bird_NOUN chirps_VERB"
]

# Initialize
emission_counts = defaultdict(lambda: defaultdict(int))
tag_counts = defaultdict(int)

# Training: Count emission probabilities
for sentence in sentences:
    tokens = sentence.split()
    for token in tokens:
        word, tag = token.rsplit("_", 1)
        emission_counts[word][tag] += 1
        tag_counts[tag] += 1

# Normalize to probabilities (optional)
def normalize_counts(counts):
    probs = {}
    for word in counts:
        total = sum(counts[word].values())
        probs[word] = {tag: count / total for tag, count in counts[word].items()}
    return probs

# Emission probabilities
emission_probs = normalize_counts(emission_counts)

# Test set (true labels)
true_labels = [
    ["DET", "NOUN", "VERB"],
    ["DET", "NOUN", "VERB"],
    ["DET", "NOUN", "VERB"],
    ["DET", "NOUN", "VERB", "ADV"],
    ["DET", "NOUN", "VERB", "ADV"],
    ["DET", "NOUN", "VERB"],
    ["DET", "NOUN", "VERB", "ADV"],
    ["DET", "NOUN", "VERB"]
]

# Test the model and evaluate
predictions = []

for i, sentence in enumerate(sentences):
    words = sentence.split()
    predicted_tags = []

    for word in words:
        word_only = word.split('_')[0]
        #
        if word_only in emission_probs:
            predicted_tag = max(emission_probs[word_only], key=emission_probs[word_only].get)
            predicted_tags.append(predicted_tag)
        else:
            predicted_tags.append('NOUN')

    true_tags = true_labels[i]
    
    if len(predicted_tags) == len(true_tags):
        predictions.append(predicted_tags)  
    else:
        print(f"Skipping sentence {i + 1} due to tag length mismatch.")

flat_true = [tag for sublist in true_labels for tag in sublist]
flat_pred = [tag for sublist in predictions for tag in sublist]

if flat_pred:
    precision = precision_score(flat_true, flat_pred, average='weighted')
    recall = recall_score(flat_true, flat_pred, average='weighted')
    f1 = f1_score(flat_true, flat_pred, average='weighted')
    report = classification_report(flat_true, flat_pred)

    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\nClassification Report:")
    print(report)
else:
    print("No valid predictions were made.")
