# Plagiarism Detection Web Application

This web application detects plagiarism between two text documents by calculating similarity scores and comparing the input text against a predefined set of source texts. It uses Flask for the web framework and Scikit-learn for text similarity calculations.

## Features

- Upload and compare two text documents.
- Calculate similarity percentage between the documents using cosine similarity.
- Match input text with a predefined set of source texts.
- Display results with similarity scores and source matches.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework in Python.
- **Scikit-learn**: A library for machine learning and data analysis in Python, used here for text vectorization and similarity calculation.
- **HTML/CSS**: For creating the web pages.

## Project Structure

- `app.py`: The main Flask application file containing routes and logic for plagiarism detection.
- `templates/`
  - `index.html`: The HTML form for uploading documents.
  - `results.html`: The HTML page displaying similarity results and source matches.

## Installation and Setup

### 1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

### 2. **Create a Virtual Environment**
    python -m venv venv

### 3. **Activate the Virtual Environment**
### On Windows:
    venv\Scripts\activate
### On macOS/Linux:
    source venv/bin/activate

### 4. **Install Dependencies**
    pip install -r requirements.txt

### 5. **Run the Flask Application**
    python app.py

### 6. Access the Application
Open your web browser and go to http://127.0.0.1:5000 to use the application.

## Usage
 - **Upload Documents:** On the main page, upload two text documents that you want to compare.
 - **View Results:** After submission, the application will display the similarity score between the documents and any matches with the predefined sources.

## Predefined Sources
The application includes a sample set of sources for matching. You can modify SOURCE_DATABASE in app.py to include more sources or update existing ones.

## Future Enhancements
- **Advanced Matching Algorithms:** Incorporate more sophisticated algorithms for better detection of plagiarism.
 - **Database Integration:** Use a database to store and manage source texts dynamically.
 - **User Authentication:** Add user authentication for saving and managing document comparisons.
   
## Contributing
Feel free to fork the repository and submit pull requests. For bug reports or feature requests, please open an issue on the GitHub repository.

## License
This project is licensed under the **MIT License**. See the **LICENSE** file for details.
