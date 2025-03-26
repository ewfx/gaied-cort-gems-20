**Gen AI Orchestrator for Email and Document Triage/Routing**

This project processes .eml email files to:

Extract text and attachments (PDF, Word, Image).

Detect duplicate emails using sentence embeddings.

Extract key attributes dynamically using Together AI.

Classify emails into categories and subcategories.

Save results in a JSON file with classification and extracted attributes.
![image](https://github.com/user-attachments/assets/6c98a238-315c-4852-9f30-a20f2d786d94)
![image](https://github.com/user-attachments/assets/e6f223d8-03bb-4d86-b8a5-817ce41448f2)

📚 **Prerequisites**
1. Install Required Software
Python 3.9 or higher

2.Tesseract OCR (for image text extraction)

2.** **Install Required Python Packages****
Run the following command to install all necessary dependencies:
pip install -r requirements.txt


⚙️** Configuration**
1. Environment Variables

Create a .env file in the root directory and add:
Tesseract OCR Path
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

Together AI API Key
TOGETHER_AI_API_KEY=your_together_ai_api_key

Together AI API URL
TOGETHER_AI_URL=https://api.together.ai/completions
Replace your_together_ai_api_key with your actual Together AI API key.

🚀 **Running the Application**
1. Start Flask API
Run the Flask application:
python app.py
The application will run at:
http://127.0.0.1:5000/

2.UI :
Go to the code\src\EmailClassification.html and run it on live server.
The UI looks like this : ![image](https://github.com/user-attachments/assets/f1744f7a-f9ef-41dd-b59b-93bf6875ae36)



📨 Processing Emails
1. Add EML Files
Place .eml files into the eml_files folder.

2. Trigger Email Processing
Access the API endpoint to start email processing:

http://127.0.0.1:5000/process_email


The system will:

Extract email contents and attachments.

Segment emails based on date.

Detect duplicates using SentenceTransformer embeddings.

Extract key attributes using Llama-3.3-70B-Instruct-Turbo.

Classify emails into categories and subcategories.

3. View Results
The results will be saved in:
/output/email_duplicates_with_attributes.json and displayed in UI

## 🏗️ Tech Stack
- 🔹 Frontend: HTML
- 🔹 Backend: Flask Python
- 🔹 Database: 
- 🔹 Other: LLama Models

## 👥 Team : CORT_Gems_2.0
  
