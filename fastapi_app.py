import os
import base64
import cv2
import numpy as np
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set up the application
app = FastAPI()

# CORS settings
origins = ["*"]  # Allow all origins, modify as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic authentication
security = HTTPBasic()

API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")

# Initialize the model
model = YOLO("yolo11x-seg.pt")  # Load the segmentation model
names = model.model.names

# Pydantic model for request body
class ImageRequest(BaseModel):
    image_base64: str

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    # print(credentials.username, API_USERNAME)
    # print(credentials.password, API_PASSWORD)
    if str(credentials.username) != str(API_USERNAME) or str(credentials.password) != str(API_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid credentials")


#------------------------------ Cropped Original Image ---------------------#

# @app.post("/segment/")
# async def segment_image(image_request: ImageRequest, credentials: HTTPBasicCredentials = Depends(authenticate)):
#     # Decode base64 string to image
#     image_data = base64.b64decode(image_request.image_base64)
#     img_array = np.frombuffer(image_data, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#     if img is None:
#         return JSONResponse(status_code=400, content={"message": "Invalid image data."})

#     output_images = []

#     # Perform inference
#     results = model.predict(img)

#     # Process each detected object
#     if results[0].masks is not None:
#         clss = results[0].boxes.cls.cpu().tolist()
#         masks = results[0].masks.xy

#         for i, (mask, cls) in enumerate(zip(masks, clss)):
#             # Create a blank image with the same dimensions as the input image and a transparent background
#             object_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

#             # Convert the mask to an integer format for OpenCV
#             mask = np.array(mask, dtype=np.int32)

#             # Set the region inside the mask to the detected object’s pixels
#             cv2.fillPoly(object_image, [mask], color=(255, 255, 255, 255))  # Fill mask area with white in the alpha channel
#             for c in range(3):  # Apply object pixels for RGB channels
#                 object_image[:, :, c] = np.where(object_image[:, :, 3] == 255, img[:, :, c], 0)

#             # Encode the object image to base64
#             _, buffer = cv2.imencode('.png', object_image)
#             object_image_base64 = base64.b64encode(buffer).decode('utf-8')
#             output_images.append(object_image_base64)

#     return {"output_images": output_images}

#------------------------------ Red Masked Image ---------------------#
@app.post("/segment/")
async def segment_image(image_request: ImageRequest, credentials: HTTPBasicCredentials = Depends(authenticate)):
    # Decode base64 string to image
    image_data = base64.b64decode(image_request.image_base64)
    img_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"message": "Invalid image data."})

    output_images = []

    # Perform inference
    results = model.predict(img)

    # Define the color for the mask (RGB format)
    mask_color = (255, 0, 0)  # Red color

    # Process each detected object
    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy

        for i, (mask, cls) in enumerate(zip(masks, clss)):
            # Create a blank image with an alpha channel
            object_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

            # Convert the mask to an integer format for OpenCV
            mask = np.array(mask, dtype=np.int32)

            # Set the region inside the mask to the specified color (red) and set alpha to 255 (opaque)
            cv2.fillPoly(object_image, [mask], color=(mask_color[2], mask_color[1], mask_color[0], 255))  # OpenCV uses BGR

            # Set the alpha channel to 0 (transparent) for pixels outside the mask
            object_image[:, :, 3] = np.where(object_image[:, :, 3] == 255, 255, 0)

            # Encode the color-coded mask image to base64
            _, buffer = cv2.imencode('.png', object_image)
            object_image_base64 = base64.b64encode(buffer).decode('utf-8')
            output_images.append(object_image_base64)

    return {"output_images": output_images}

# Run the application with `uvicorn`:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
