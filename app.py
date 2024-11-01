# run_both.py
import subprocess

def run_fastapi():
    subprocess.run(["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"])

def run_gradio():
    subprocess.run(["python", "gradio_app.py"])

if __name__ == "__main__":
    import threading

    # Create threads for FastAPI and Gradio
    fastapi_thread = threading.Thread(target=run_fastapi)
    gradio_thread = threading.Thread(target=run_gradio)

    fastapi_thread.start()
    gradio_thread.start()

    fastapi_thread.join()
    gradio_thread.join()
