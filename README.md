# Auto NER Annotation with Hugging Face Model

An automated Named Entity Recognition (NER) annotation service that leverages Hugging Face transformer models to generate annotations for text data. This service is designed to streamline the NER annotation process by automatically identifying and labeling entities in text.

## 🌟 Features

- **Automated NER Processing**: Utilizes state-of-the-art Hugging Face transformer models for entity recognition
- **Flexible Input Handling**: Accepts both JSON file uploads and direct JSON data
- **Dynamic Model Loading**: Supports multiple models with automatic latest model selection
- **Structured Output**: Generates annotations in a standardized format compatible with common annotation tools
- **RESTful API**: Easy-to-use HTTP endpoint for inference
- **Model Management**: Support for multiple model versions with automatic latest model detection
- **Coming Soon**: Interactive frontend interface for visualization and annotation management

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Dependencies (specified versions):
  ```
  Flask==3.0.3
  transformers==4.45.2
  Werkzeug==3.0.4
  ```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Auto-NER-Annotation-with-Hugging-Face-Model.git
cd Auto-NER-Annotation-with-Hugging-Face-Model
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your models directory:
```bash
mkdir models
# Place your Hugging Face NER models in the models directory
```

## 📋 Usage

### Starting the Server

Run the Flask application:
```bash
python app.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### POST /inference

Submit text for NER annotation processing.

**Input Formats:**

1. JSON File Upload:
```json
{
    "Objects": [
        {
            "Text": "Your text content here"
        }
    ]
}
```

2. Direct JSON Request:
```json
{
    "Objects": [
        {
            "Text": "Your text content here"
        }
    ]
}
```

**Optional Parameters:**
- `model_path`: Specify a particular model version (defaults to latest)

**Example cURL Request:**
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"Objects": [{"Text": "John works at Microsoft in Seattle"}]}'
```

### Output Format

The service returns annotations in the following structure:
```json
[
    {
        "id": 1,
        "annotations": [
            {
                "completed_by": 1,
                "result": [
                    {
                        "value": {
                            "start": 0,
                            "end": 4,
                            "text": "John",
                            "labels": ["PERSON"]
                        },
                        "id": "label_uuid",
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "origin": "manual"
                    }
                ]
            }
        ],
        "data": {
            "text": "John works at Microsoft in Seattle"
        }
    }
]
```

## 🔄 Upcoming Features

- **Interactive Frontend Interface**:
  - Real-time annotation visualization
  - Annotation editing capabilities
  - Model performance metrics
  - Batch processing interface
  - User authentication and project management

- **Enhanced Backend Capabilities**:
  - Support for custom model training
  - Batch processing optimization
  - Enhanced error handling and validation
  - API authentication
  - Result caching and optimization

## 🛠️ Project Structure

```
Auto-NER-Annotation-with-Hugging-Face-Model/
├── app.py              # Main Flask application
├── models/            # Directory for NER models
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## 💡 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Acknowledgments

- Hugging Face for their excellent transformers library
- Flask framework for the web application structure
- Contributors and maintainers of the dependencies used in this project
