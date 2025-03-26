import os
import json
import uuid
import hashlib
import requests
import re
import pandas as pd
import PyPDF2
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser
from docx import Document
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import numpy as np
from datetime import datetime
from collections import OrderedDict
import warnings
warnings.simplefilter(action='ignore')


# --- Load environment variables ---
load_dotenv()

# --- Configurations ---
app = Flask(__name__)
CORS(app)

# API & File Paths
#76ea4d533633583fdf0f32219f4ac2855cbdbeaf75cff7205aa843414ecff580
#e33a693e8c688fe209345cf986c085de9c30335cdc3567e885e5d30bc5b921af

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
api_key = os.getenv("TOGETHER_AI_API_KEY", "4bbba4d7aeac8661a4ef221430dda686b99922938e79ffddb78ccf953beac70d")
url = os.getenv("TOGETHER_AI_URL", "https://api.together.ai/completions")
email_folder = os.path.join(os.path.dirname(__file__), "extract")
categories_path = os.path.join(os.path.dirname(__file__), "categories_data.txt")

# Model Initialization
model = SentenceTransformer("all-MiniLM-L6-v2")

# Duplicate Hashes
duplicate_hashes = set()


# --- Utility Functions ---
def sanitize_filename(name):
    """Sanitize filenames to avoid invalid characters."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def get_unique_filename(folder, filename):
    """Generate a unique filename to avoid overwrite."""
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename
    while os.path.exists(os.path.join(folder, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1
    return unique_filename


def save_text_to_file(text, filename, folder):
    """Save extracted text to a file."""
    filepath = os.path.join(folder, get_unique_filename(folder, filename))
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(text.strip())
    print(f"Saved: {filepath}")


def create_extraction_folder(subject, email_guid, email_filename):
    """Create folder structure for extracted data."""
    clean_subject = sanitize_filename(subject)[:50]
    base_folder = os.path.join(email_folder, f"{email_guid}_{email_filename}")
    extract_data_folder = os.path.join(base_folder, "extract_data")
    os.makedirs(extract_data_folder, exist_ok=True)
    return base_folder, extract_data_folder


# --- Email Extraction Functions ---
def extract_email_text(msg, extract_data_folder, email_guid):
    """Extract and save email content."""
    subject = msg.get("Subject", "No Subject")
    date = msg.get("Date", "No Date")
    email_body =f"Date: {date}\n\n"
    email_body += f"Subject: {subject}\n\n"

    for part in msg.walk():
        content_type = part.get_content_type()
        charset = part.get_content_charset() or "utf-8"
        if content_type == "text/plain":
            email_body += part.get_payload(decode=True).decode(charset, errors="ignore")
        elif content_type == "text/html":
            html_content = part.get_payload(decode=True).decode(charset, errors="ignore")
            email_body += BeautifulSoup(html_content, "html.parser").get_text()

    save_text_to_file(email_body, f"{email_guid}_email_content.txt", extract_data_folder)


def extract_attachments(msg, extract_folder, extract_data_folder, email_guid):
    """Extract attachments (PDFs, Word, Images, EML)."""
    for part in msg.walk():
        filename = part.get_filename()
        content_type = part.get_content_type()
        if filename:
            save_attachment(part, extract_folder, extract_data_folder, email_guid, filename, content_type)


def save_attachment(part, extract_folder, extract_data_folder, email_guid, filename, content_type):
    """Save and process different types of attachments."""
    attachment_path = os.path.join(extract_folder, get_unique_filename(extract_folder, f"{email_guid}_{sanitize_filename(filename)}"))

    with open(attachment_path, "wb") as file:
        file.write(part.get_payload(decode=True))

    print(f"Attachment saved: {attachment_path}")

    if filename.lower().endswith((".pdf")):
        extract_pdf_text(attachment_path, extract_data_folder, email_guid)
    elif filename.lower().endswith((".doc", ".docx")):
        extract_word_text(attachment_path, extract_data_folder, email_guid)
    elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
        extract_image_text(attachment_path, extract_data_folder, email_guid)


def extract_pdf_text(pdf_path, extract_data_folder, email_guid):
    """Extract and save text from PDF."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            content = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        save_text_to_file(content, f"{email_guid}_pdf.txt", extract_data_folder)
    except Exception as e:
        print(f"Error extracting PDF: {e}")


def extract_word_text(word_path, extract_data_folder, email_guid):
    """Extract and save text from Word documents."""
    try:
        doc = Document(word_path)
        content = "\n".join(para.text for para in doc.paragraphs)
        save_text_to_file(content, f"{email_guid}_word.txt", extract_data_folder)
    except Exception as e:
        print(f"Error extracting Word text: {e}")


def extract_image_text(image_path, extract_data_folder, email_guid):
    """Extract and save text from image using OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        save_text_to_file(text, f"{email_guid}_image_text.txt", extract_data_folder)
    except Exception as e:
        print(f"Error extracting image text: {e}")


# --- Email Processing Functions ---
def extract_eml_data(eml_file_path):
    """Process and extract all data from .eml file."""
    email_guid = str(uuid.uuid4())
    try:
        with open(eml_file_path, "rb") as eml_file:
            msg = BytesParser(policy=policy.default).parse(eml_file)

        subject = msg.get("Subject", "No Subject")
        email_filename = os.path.basename(eml_file_path)
        extract_folder, extract_data_folder = create_extraction_folder(subject, email_guid, email_filename)

        print(f"Processing Email - Subject: {subject} | GUID: {email_guid}")
        extract_email_text(msg, extract_data_folder, email_guid)
        extract_attachments(msg, extract_folder, extract_data_folder, email_guid)
    except Exception as e:
        print(f"Error processing email: {e}")


def process_eml_folder(folder_path):
    """Process all .eml files in a folder."""
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".eml"):
                eml_file_path = os.path.join(root, filename)
                print(f"Processing: {eml_file_path}")
                extract_eml_data(eml_file_path)


# --- Classification and Attribute Extraction ---
def get_key_attributes(email_content):
    """Extract key attributes dynamically using Together AI."""
    prompt = f"""
    Extract the key attributes from the following email.Don't extract any email headers like content type,mime version etc.Dynamically identify the attributes based on the email content and provide the output in the format:

    Key Attributes:
      attribute_name_1: value
      attribute_name_2: value
      attribute_name_3: value
      ...

    Email: "{email_content}"

    Output:
    Key Attributes:
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "prompt": prompt,
        "max_tokens": 300,
        "top_p": 0.9,
        "n": 1,  # Single response
        "stop": ["\n\n", "###", "```", "Explanation:", "Code:", "Advice:"]  # Stop conditions to prevent extra content
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()["choices"][0]["text"].strip()
        return result
    except requests.RequestException as e:
        print(f"Error extracting key attributes: {e}")
        return "No attributes extracted"


def classify_text_with_togetherai(text, categories):
    """Classify text using Together AI API."""
    
    stop_sequences = ["\n\n", "###", "```", "'''", "def", "Explanation:", "Code:", "Advice:"]
    headers = {"Authorization": "Bearer fefa5c98389cbc93862666732870431708e713249286aaea887e5ee011ef2695", "Content-Type": "application/json"}
    
    prompt = f"""
    Classify the following text into one of the categories and the sub-category within the selected category. Explain the reasoning.
    Text: {text}
    Category and its sub-categories: {categories}

    Category: [category]
    Sub-Category: [sub-category]
    Reasoning: [reasoning]
    """
    
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "prompt": prompt,
        "max_tokens": 250,
        "top_p": 0.8,
        "stop": stop_sequences,
        "logprobs": 5,
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()        
        result = response.json()["choices"][0]["text"].strip()
        confidence_score = (get_confidence_score(response.json()))

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        result = "Error"
    
    return result, confidence_score

def get_confidence_score(response_data):
    if "choices" in response_data and response_data["choices"] and "logprobs" in response_data["choices"][0]:
        logprobs = response_data["choices"][0]["logprobs"]["token_logprobs"]
        if logprobs and all(lp is not None for lp in logprobs):
            # A simple way is to take the average of the log probabilities
            avg_logprob = np.mean(logprobs)
            return round(float(np.exp(avg_logprob)) * 100, 6)
    return None

def load_email_contents(email_folder):
    """Load and return email content from folder."""
    emails = {}
    for root, _, files in os.walk(email_folder):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    emails[filename] = file.read()
    return emails


def load_categories():
    """Load categories from categories_data.txt."""
    with open(categories_path, "r") as f:
        return f.read()

def mail_segmentation(email_chain):
    """Segments email chain into key-value pairs of date and email content."""
    
    # --- Date pattern to capture RFC 2822 dates ---
    date_pattern = r"(?P<date>(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}\s\d{2}:\d{2}:\d{2})"
    
    # --- Dictionary to hold segmented emails ---
    segments = {}
    
    # --- Find all date matches and positions ---
    mail_dates = []
    for match in re.finditer(date_pattern, email_chain):
        mail_dates.append((match.group("date"), match.start()))

    # --- If no dates found, return empty ---
    if not mail_dates:
        print("No valid dates found in the email chain!")
        return OrderedDict()

    # --- Loop through identified email segments ---
    for i, (date_str, start_idx) in enumerate(mail_dates):
        # Define segment range
        end_idx = mail_dates[i + 1][1] if i + 1 < len(mail_dates) else len(email_chain)

        # Extract and clean email content
        email_content = email_chain[start_idx:end_idx].strip()

        # Convert date string to standard format
        try:
            datetime_obj = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S")
            formatted_date = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        # Add to dictionary
        segments[formatted_date] = email_content

    # --- Return segments sorted in reverse order (latest first) ---
    return OrderedDict(sorted(segments.items(), reverse=True))


# --- Main API Endpoint ---
@app.route("/process_email", methods=["GET"])
def api():
    """Main API to process emails, check duplicates, and classify."""
    folder_path = os.path.join(os.path.dirname(__file__), "eml_files")
    process_eml_folder(folder_path)

    emails_dict = load_email_contents(email_folder)
    categories = load_categories()
    df = pd.DataFrame(columns=["email_name", "is_duplicate","request_type", "sub_request_type", "key_attributes",  "reasoning", "confidence_score"])
    unique_emails = []

    for email_name, email_content in emails_dict.items():
        # --- Check for Duplicate ---
        is_duplicate, similarity_score = False, 0.0
        email_embedding = model.encode(email_content, convert_to_tensor=True)

        for unique_email in unique_emails:
            similarity_score = util.pytorch_cos_sim(email_embedding, unique_email).item()
            if similarity_score >= 0.9:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_emails.append(email_embedding)
        
        segmented_mails = mail_segmentation(email_content)

        actual_email_content = None
        actual_key_attributes = None

        # Iterate through segmented mails to find the first valid key attributes
        for segment in segmented_mails.values():
            email_content = segment
            key_attributes = get_key_attributes(email_content)

            # Check if key_attributes is not equal to "No attributes extracted."
            if key_attributes != "No attributes extracted.":
                actual_key_attributes = key_attributes
                actual_email_content = email_content
                break

        # If no valid email is found, return None or any fallback value
        if not actual_email_content:
            actual_email_content = "No valid email content found."

            # --- Check for Duplicate ---

        # --- Get Key Attributes ---
        

        # --- Classify Email ---
        classification_result, confidence_score = classify_text_with_togetherai(email_content, categories)
        classification_result = classification_result.replace("  ", "").replace(" Category:", "Category:").replace(
                " Sub-Category:", "Sub-Category:").replace(" Reasoning:", "Reasoning:")
        # Updated regex to match without square brackets
        category_match = re.search(r"Category:\s*(.*)", classification_result)
        sub_category_match = re.search(r"Sub-Category:\s*(.*)", classification_result)
        reasoning_match = re.search(r"Reasoning:\s*(.*)", classification_result)

        # Extracting values
        category = category_match.group(1).strip() if category_match else "Unknown"
        sub_category = sub_category_match.group(1).strip() if sub_category_match else "Unknown"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No Reasoning"

        # --- Add to DataFrame ---
        # --- Clean and Normalize Data ---
        # Remove path prefix from email name
        cleaned_name = email_name.lower().replace("d:\\email_classification\\extract\\", "").replace("\\", "/").split(".eml")[0]
        cleaned_key_attributes = re.sub(r"[\\\n]+", " ", actual_key_attributes).replace("```", "").strip()

# Remove any unnecessary quotes and backslashes in reasoning
        cleaned_reasoning = reasoning.replace("\\", "").replace('"', "").replace("\n", " ").strip()
        pd.DataFrame(columns=["email_name", "is_duplicate","request_type", "sub_request_type", "key_attributes",  "reasoning", "confidence_score"])
    
        # --- Add Cleaned Data to DataFrame ---
        df = pd.concat([df, pd.DataFrame({
            "email_name": [cleaned_name],
            "is_duplicate": [is_duplicate],
            "request_type": [category],
            "sub_request_type": [sub_category],
            "key_attributes": [cleaned_key_attributes],
            "reasoning": [cleaned_reasoning],
            "confidence_score": [confidence_score]  # Ensure confidence score is included
        })], ignore_index=True)


    # --- Save Output ---
    json_data = {"email_duplicates_with_attributes": df.to_dict(orient="records")}
    # Define output folder and file path
    output_folder = os.path.join(os.path.dirname(__file__), "output")
    output_path = os.path.join(output_folder, "email_duplicates_with_attributes.json")

    # Create the 'output' folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Write the JSON data to the file
    with open(output_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Duplicate check and classification completed. Results saved in {output_path}")
    return jsonify(json_data)


if __name__ == "__main__":
    app.run(debug=True)
