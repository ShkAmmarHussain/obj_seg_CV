import os
import base64
import requests
from io import BytesIO
from PIL import Image
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# API configuration
API_URL = "http://localhost:8000/segment/"  # Adjust this if your FastAPI app runs elsewhere
API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")

def segment_image(image):
    # Convert the NumPy image to PIL Image
    pil_image = Image.fromarray(image)

    # Encode image to base64
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # Change format as needed
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Prepare the payload
    payload = {
        "image_base64": image_base64
    }

    # Send the request to the FastAPI endpoint
    response = requests.post(API_URL, json=payload, auth=(API_USERNAME, API_PASSWORD))
    
    if response.status_code == 200:
        output = response.json()
        # Decode the base64 images to PIL Images
        output_images = []
        for img_base64 in output["output_images"]:
            img_data = base64.b64decode(img_base64)
            img = Image.open(BytesIO(img_data))
            output_images.append(img)
        return output_images
    else:
        return f"Error: {response.text}"

def process_image(image):
    # Process the uploaded image and return the segmented images
    return segment_image(image)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Object Segmentation with YOLO")
    gr.Markdown("Upload an image to segment objects with a transparent background.")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload Image")
        submit_button = gr.Button("Segment Image")

    output_gallery = gr.Gallery(label="Segmented Images", show_label=True)

    submit_button.click(fn=process_image, inputs=image_input, outputs=output_gallery)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8001, share=False)
