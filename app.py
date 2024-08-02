from flask import Flask, request, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF for PDF processing
import docx  # python-docx for DOCX processing
import re
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}
MAX_PAGES = 10
MAX_WORDS = 2000

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(filepath, max_pages=MAX_PAGES):
    """Extracts text from a PDF file, processing a limited number of pages."""
    text = ""
    with fitz.open(filepath) as doc:
        page_count = min(max_pages, len(doc))
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text, page_count

def extract_text_from_docx(filepath):
    """Extracts text from a DOCX file."""
    doc = docx.Document(filepath)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(filepath):
    """Extracts text from a TXT file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def get_text(filepath):
    """Extracts text based on the file extension and checks page and word limits."""
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        text, page_count = extract_text_from_pdf(filepath)
        if page_count > MAX_PAGES:
            return None, "maximum pages 10 exceeds"
    elif ext == 'docx':
        text = extract_text_from_docx(filepath)
    elif ext == 'txt':
        text = extract_text_from_txt(filepath)
    else:
        raise ValueError("Unsupported file format.")
    
    words = extract_words(text)
    if len(words) > MAX_WORDS:
        return None, "maximum words 2000 exceeds"

    return text, None

def extract_words(text):
    """Extracts words from a text."""
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def calculate_hash(text):
    """Calculate hash for text."""
    return hashlib.md5(text.encode()).hexdigest()

def find_exact_matches(words1, words2):
    """Find exact matches of word sequences."""
    matches = set()
    len_words1 = len(words1)
    len_words2 = len(words2)
    
    for i in range(len_words1):
        for j in range(i + 1, len_words1 + 1):
            segment = tuple(words1[i:j])
            if segment in [tuple(words2[k:k + len(segment)]) for k in range(len_words2 - len(segment) + 1)]:
                matches.add(segment)
    
    return matches

def find_fingerprints(text, fragment_size=5):
    """Find fingerprints using text fragments."""
    words = extract_words(text)
    fingerprints = set()
    for i in range(len(words) - fragment_size + 1):
        fragment = ' '.join(words[i:i + fragment_size])
        fingerprints.add(calculate_hash(fragment))
    return fingerprints

def vectorize_text(words, vocabulary):
    """Create a vector representation of the text based on a vocabulary."""
    vector = [0] * len(vocabulary)
    word_count = Counter(words)
    for idx, word in enumerate(vocabulary):
        vector[idx] = word_count[word]
    return vector

def cosine_similarity(vectorA, vectorB):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vectorA, vectorB)
    normA = np.linalg.norm(vectorA)
    normB = np.linalg.norm(vectorB)
    return dot_product / (normA * normB) if normA and normB else 0

def create_donut_chart(percentage):
    """Creates a donut chart image to represent the plagiarism percentage."""
    fig, ax = plt.subplots()
    ax.pie([percentage, 100 - percentage], colors=['red' if percentage > 80 else 'orange' if percentage > 50 else 'yellow' if percentage > 10 else 'green', 'grey'],
           startangle=90, wedgeprops=dict(width=0.3))
    ax.text(0, 0, f'{percentage:.2f}%', horizontalalignment='center', verticalalignment='center', fontsize=20, fontweight='bold')
    plt.axis('equal')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def calculate_plagiarism(text1, text2):
    """Calculates plagiarism based on various methods."""
    words1 = extract_words(text1)
    words2 = extract_words(text2)
    
    exact_matches = find_exact_matches(words1, words2)
    fingerprints1 = find_fingerprints(text1)
    fingerprints2 = find_fingerprints(text2)
    
    fingerprint_matches = fingerprints1.intersection(fingerprints2)
    
    # Create common vocabulary
    vocab = set(words1) | set(words2)
    vocab = sorted(vocab)  # Sort to ensure consistent ordering
    
    vectorA = vectorize_text(words1, vocab)
    vectorB = vectorize_text(words2, vocab)
    
    cosine_sim = cosine_similarity(
        np.array(vectorA),
        np.array(vectorB)
    )
    
    exact_match_ratio = len(exact_matches) / max(len(words1), len(words2)) * 100
    fingerprint_match_ratio = len(fingerprint_matches) / max(len(fingerprints1), len(fingerprints2)) * 100
    
    plagiarism_percentage = max(exact_match_ratio, fingerprint_match_ratio, cosine_sim * 100)
    return round(plagiarism_percentage)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'file1', filename1)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'file2', filename2)
            file1.save(file1_path)
            file2.save(file2_path)

            text1, error1 = get_text(file1_path)
            text2, error2 = get_text(file2_path)

            if error1:
                return render_template('index.html', error_message=error1)
            if error2:
                return render_template('index.html', error_message=error2)

            plagiarism_percentage = calculate_plagiarism(text1, text2)
            donut_chart = create_donut_chart(plagiarism_percentage)
            return render_template('result.html', percentage=plagiarism_percentage, donut_chart=donut_chart)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'file1'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'file2'), exist_ok=True)
    app.run(debug=True)
