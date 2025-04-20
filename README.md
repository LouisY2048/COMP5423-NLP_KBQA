# Knowledge Base Question Answering System

A Knowledge Base Question Answering (KBQA) system based on the NQ10K dataset. This system search for relevant documents from the knowledge base and uses a large language model to generate answers.

## Project Structure

```
/
|---data/              # Dataset storage
|---backend/           # Backend code
|   |---utils/         # Utility functions
|   |---retrieval/     # Document retrieval module
|   |---generation/    # Answer generation module
|   |---models/        # Model files
|   |---main.py        # Backend entry point
|   |---evaluation.py  # Evaluation script
|---frontend/          # Frontend code
|   |---src/           # Frontend source code
|   |---public/        # Public resources
```

## System Features

The system implements three main features:

1. **Document Retrieval**
   - Keyword-based retrieval using TF-IDF algorithm
   - Vector space retrieval using Sentence-Transformer

2. **Answer Generation**
   - Utilizes Qwen2.5-7B-Instruct large language model
   - Generates answers based on retrieved documents
   - Implements carefully designed prompt templates

3. **User Interface**
   - Intuitive interface for question input
   - Preset question suggestions
   - Displays retrieved documents and generated answers
   - Responsive design with modern UI elements

## Installation and Setup

### Prerequisites

- Python 3.11.x
- Node.js 14.0 or higher
- npm 6.0 or higher

### Backend Setup

1. Create and activate a virtual environment (recommended):

2. Install Python dependencies(Please ensure that you are in the virtual envionment):
```bash
cd backend
pip install -r requirements.txt
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

## Running the System

### Backend Server

1. Start the backend server:
```bash
cd backend
python main.py
```
The server will run on `http://localhost:8000`

### Frontend Development

1. Start the development server(Create a new command line terminal to finish this step!!!):
```bash
cd frontend
npm run serve
```
The frontend will be available at `http://localhost:3000`


## Please note that there is already test_predict.jsonl in the project and it is in the root directory. If you do not need to regenerate test_predict.jsonl, please ignore the following sections.
## Important Notes Before Create Test_predict File From 0 to 1

To prepare for creating test_predict file, please:

1. Data Preprocessing:
   -Document preprocessing:
```bash
python backend/utils/doc_modifier.py
```

2. Train model and save relavant parameters
   -For keyword retrieval solution:
```bash
python backend/train_model.py --model keyword
```
   -For dense retrieval solution:
```bash
python backend/train_model.py --model dense
```
3. Predict for test data
```bash
python backend/create_test_file.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.