import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Job roles dataset (items in our recommendation engine)
job_roles = {
    "Data Scientist": "python sql machine learning data analysis statistics",
    "DevOps Engineer": "aws docker kubernetes ci cd automation cloud",
    "Backend Developer": "java python sql apis databases rest",
    "Frontend Developer": "javascript react css html ui design",
    "Cloud Architect": "aws azure cloud infrastructure networking security"
}

df = pd.DataFrame(list(job_roles.items()), columns=["role", "skills"])
print("Available Job Roles:\n", df, "\n")

# Step 2: Take user input (minimum 3 skills)
user_input = input("Enter your skills (comma separated, e.g. python, cloud, automation): ")
user_skills = [skill.strip().lower() for skill in user_input.split(",")]

# Step 3: Combine job skills + user skills into one shared vocabulary space
documents = df["skills"].tolist()
documents.append(" ".join(user_skills))

# Step 4: TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 5: Cosine similarity between user vector and job role vectors
user_vector = tfidf_matrix[-1]
job_vectors = tfidf_matrix[:-1]
similarity_scores = cosine_similarity(user_vector, job_vectors)

# Step 6: Sort and filter -> Top 3 recommendations
df["similarity"] = similarity_scores[0]
top_matches = df.sort_values(by="similarity", ascending=False).head(3)

print("\nTop 3 Recommended Career Paths for you:")
print(top_matches[["role", "similarity"]].to_string(index=False))