# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.feature_extraction.text import TfidfVectorizer


# Loading dataset
df = pd.read_csv("merged ai and job market.csv")


# Removing missign or 'unknown' skills 
df = df[df["skills_standardized"].notna()]
df = df[df["skills_standardized"] != "unknown"]

# Splitting skills into a list
df["skills_list"] = df["skills_standardized"].apply(lambda x: x.split(";"))

# Frequency analysis
all_skills = [skill for sublist in df["skills_list"] for skill in sublist]

# Count frequency
skill_counts = pd.Series(all_skills).value_counts()

# Get unique skills
unique_skills = set(all_skills)
print("Total unique skills:", len(unique_skills))
print(sorted(unique_skills))

# Top N skills
top_n = 5
top_skills = skill_counts.head(top_n)

print("Top Skills:\n", top_skills)


# Visualization
top_n = 5
top_skills = skill_counts.head(top_n)

plt.figure(figsize=(8,5))

# Create gradient colors
colors = cm.Blues(np.linspace(0.9, 0.4, len(top_skills)))

bars = plt.bar(top_skills.index, top_skills.values, color=colors)

# Add labels above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        height,                             
        f"{int(height):,}",                 
        ha='center',
        va='bottom'
    )

plt.xlabel("Skills", fontweight='bold')
plt.ylabel("Frequency", fontweight='bold')
plt.title("Top Skills in Tech Job Postings")

plt.tight_layout()
plt.show()


# TF-IDF for recommendation system
df["skills_text"] = df["skills_list"].apply(lambda x: " ".join(x))

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["skills_text"])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)