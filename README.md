# Amazon Review Retrieval Setup Tutorial

## Prerequisites
- Ensure that Python 3.11 is installed on your system.

## Steps to Start the UI

### 1. Create a Virtual Environment

```bash
python3.11 -m venv venv
```

Activate the virtual environment:
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Run `nltk_download.py`

```bash
python nltk_download.py
```

### 4. Download the Dataset
Download the Dataset from https://www.kaggle.com/datasets/marcelzisser/amazon-reviews-sentiment-analysis and place it into a folder named "data". Rename the dataset to data.csv.

### 5. Run `ui.py`

```bash
python ui.py
```

### 6. Access the UI
After running `ui.py`, wait for the debugger to activate. Once the debugger is active, an IP address will be printed to the terminal. Open this address in your browser to access the UI.

