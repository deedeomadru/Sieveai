from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import textract
import re
import string
from pyresparser import ResumeParser
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure NLTK dependencies are downloaded
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
            r"/*": {"origins": ["http://localhost:5000"]}})

# Set upload folder for resumes
app.config['UPLOAD_FOLDER'] = os.path.join('..', 'uploads')

######################################

###### NLP MODEL SECTION #############

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def extract_text_from_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    text = textract.process(filepath)
    return text.decode('utf-8')

def rank_resumes(job_description, resume_files):
    job_description = preprocess_text(job_description)
    resume_texts = [preprocess_text(extract_text_from_file(file)) for file in resume_files]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_description] + resume_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    resume_rankings = sorted(zip(resume_files, similarity_scores), key=lambda x: x[1], reverse=True)
    return resume_rankings

######################################


###### FINAL PROCESS SECTION #########

@app.route('/process', methods=['POST'])
def show_result():
    data = request.get_json()
    my_profile = data['profile']
    user_id = data['userId']
    resumes = data['resumes']
    my_tags = data['tags']
    my_jd = data['jd']

    # Only process resumes that match the job profile
    filtered_files = []
    for resume in resumes:
        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], user_id, resume)
        # Optionally: Add logic here to filter based on specific criteria
        filtered_files.append(resume_path)

    # Rank the filtered resumes
    rankings = rank_resumes(my_jd, filtered_files)

    # Gather additional data if needed (e.g., using ResumeParser)
    res = []
    for file, score in rankings:
        user_info = ResumeParser(file).get_extracted_data()
        res.append({
            'resumeId': os.path.basename(file),
            'score': round(score * 100, 2),
            'userInfo': user_info
        })

    return jsonify(res)

######################################


@app.route('/')
def hello_world():
    return "Hello World"


if __name__ == '__main__':
    app.run(port=5002)
