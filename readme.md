## Project Structure

```
├── data/                      # Data files (e.g., PDFs)
│   └── deal_book.pdf
├── app.py                    # FastAPI app and routes
├── .gitignore                 # gitignore file
├── models.py                  # Pydantic models and state definitions
├── origin_main.py             # Original and combined py file
├── main_with_rc.py            # origin_main but with relevant context as output
├── config.py                  # Environment variables and configuration
├── requirements.txt           # Project dependencies
├── .env                       # Environment variables
├── services.py                # Core logic (LLM, vector store, graph, etc.)
└── README.md                  # Project documentation

```
