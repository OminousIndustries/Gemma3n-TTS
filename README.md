# Gemma 3n Multi-Modal Chatbot Starter Kit

This repository contains the basic code for a multi-modal (voice, image, and text) chatbot application. It features a React-based frontend and a Python-based backend that serves a local Gemma 3n model for inference.

![Gemma 3n Chatbot Demo](Image.png)

This project is intended as a self-starter kit for developers. The instructions below provide a high-level overview of how to get the application running.

## Project Structure

This project is composed of a backend and a frontend. The file structure is as follows:

-   `gemma_server.py`: The FastAPI backend that exposes endpoints for the Gemma model.
-   `gemma_record_gui.py`: A helper script that contains model loading and utility functions used by the server.
-   `App.jsx`: The main React component for the frontend user interface.
-   `Image.png`: The application preview image displayed above.

## Getting Started

This guide assumes you have a working knowledge of Python and JavaScript development environments.

### Backend Setup (Python / FastAPI)

1.  **Place Files**: Ensure `gemma_server.py` and `gemma_record_gui.py` are in the same directory. The server imports logic directly from the GUI script.

2.  **Hugging Face Access**: The `google/gemma-3n-e4b-it` model is gated. You must first visit the [model page on Hugging Face](https://huggingface.co/google/gemma-3n-e4b-it), accept the license terms, and log in to your Hugging Face account from your terminal:
    ```bash
    pip install huggingface_hub
    huggingface-cli login
    ```

3.  **Install Dependencies**: It is highly recommended to use a Python virtual environment. Install the required libraries using pip:
    ```bash
    pip install "fastapi[all]" torch transformers scipy sounddevice numpy accelerate python-multipart
    ```

4.  **Run the Server**: Launch the backend server from your terminal. It will be accessible at `http://localhost:8000`.
    ```bash
    uvicorn gemma_server:app --reload
    ```
    The first time you run this, the script will download the Gemma model, which may take some time.

### Frontend Setup (React)

1.  **Scaffold a React App**: Use a tool like Vite to create a new React project.
    ```bash
    npm create vite@latest gemma-chatbot-ui -- --template react
    cd gemma-chatbot-ui
    ```

2.  **Add Code**: Replace the contents of the generated `src/App.jsx` file with the code from the `App.jsx` script provided.

3.  **Styling**: The UI is styled with [Tailwind CSS](https://tailwindcss.com/). You will need to install and configure it for your project.

4.  **Install Dependencies & Run**:
    ```bash
    npm install
    npm run dev
    ```
    The application will be running at `http://localhost:5173`.

---

This project is designed to provide a functional baseline. For detailed guidance on setting up specific tools like Python virtual environments, Node.js, or Tailwind CSS, ask an AI assistant for step-by-step instructions tailored to your operating system.
