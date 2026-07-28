# Invoice Scraper Backend

FastAPI backend utility for multilingual invoice table extraction and Excel sheet generation.

## Features
- Dynamic table extraction from invoices/bills using Google Gemini Vision
- MongoDB extraction history persistence
- Excel spreadsheet (.xlsx) generation with custom formatting
- CORS & production ready for deployment (Render / Heroku)

## Setup & Run
```bash
pip install -r requirement.txt
python -m uvicorn server:app --reload --port 8000
```
