# Llava REST API Server

**Author:** Pedro Afonso Dias

This repository provides a REST API for interacting with a **Llava model** using the `transformers` library and `Flask-RESTful`. The API allows users to send text and image inputs to the Llava model and receive generated responses.

## 🚀 Features
- **Llava model integration** via `transformers`
- **Flask-RESTful API** with endpoints for processing text and images
- **Configurable** via `config.json`
- **CUDA support** for GPU acceleration

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/vlm-api-server.git
cd vlm-api-server
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

Ensure you have `transformers`, `flask`, `flask-restful`, and `torch` installed.

### 4️⃣ Download the Model
The model is automatically downloaded when the API starts. However, you can manually download it using:
```python
from transformers import LlavaNextForConditionalGeneration
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf")
```

> **_NOTE:_**  Ensure you have at least 75GB of free space for the model.

---

## ⚙️ Configuration
The API uses a `config.json` file to load settings. Ensure it is in the **same directory** as `app.py`.

### Example `config.json`:
```json
{
    "model_name": "llava-hf/llava-v1.6-34b-hf",
    "port": 5000
}
```

---

## 🚀 Running the API Server

Start the API server using:
```bash
python3 app.py
```

By default, the server runs on `http://0.0.0.0:5000`. You can change the port in `config.json`.

---

## 📡 API Endpoints

### 🔹 **Health Check**
#### `GET /`
**Description:** Check if the API is running.
```json
{
    "message": "Llava API is working."
}
```

### 🔹 **Chat with the Model**
#### `POST /chat`
**Description:** Send a text input (and optional images) to the Llava model.

**Request Body (JSON):**
```json
{
    "text": "What is the capital of France?",
    "images": null
}
```

**Response (JSON):**
```json
{
    "response": "The capital of France is Paris."
}
```

---

## 🛠️ Deployment
For production, run the server using **Gunicorn**:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Or deploy using **Docker**:
```bash
docker build -t llava-api .
docker run -p 5000:5000 llava-api
```

