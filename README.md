# POS Tagger API

This project is a POS (Part-of-Speech) Tagger API built using FastAPI. It supports multiple models for POS tagging, including Keras, HMM, and a custom model.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/m2i-duo/pos-tagger-api.git
    cd pos-tagger-api
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

### Model Configuration

The model configuration is defined in `app/config/model_config.py`. You can adjust parameters such as batch size, epochs, learning rate, etc.

### API Configuration

The API configuration is defined in `app/config/api_config.py`. You can set the API title, version, description, etc.

## Running the API

To run the FastAPI server, execute the following command:
```sh
uvicorn app.main:app --reload