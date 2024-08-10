from fastapi import APIRouter, UploadFile, File, status, HTTPException
from PIL import Image
import io
import os
import numpy as np
import uuid
# Make sure this is correctly implemented
from utils.classes_names import classes
from fastapi.responses import JSONResponse
# functions calling
from utils.model_load import load_keras_model
from utils.image_preprocessing import preprocess_image

# Load the model once at startup
model = load_keras_model("model.h5")

# Create output directory if it doesn't exist
output_dir = "Files"
os.makedirs(output_dir, exist_ok=True)

router = APIRouter(
    prefix="/predict",
    tags=["Image Classification"]
)


@router.post("/")
async def get_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Image file required (jpg, png, jpeg)")

    # Read the file
    file_data = await file.read()

    try:
        # Open the image
        image_uploaded = Image.open(io.BytesIO(file_data))

        # Generate a unique file name with extension (if you plan to save it)
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"

        # If you want to save the image, uncomment the following lines
        file_location = os.path.join(output_dir, unique_filename)
        image_uploaded.save(file_location)

        # Preprocess the image
        img_array = preprocess_image(image_uploaded)

        # Make a prediction
        prediction = model.predict(img_array)

        # Display the prediction
        predicted_index = np.argmax(prediction[0])
        predicted_class = classes[predicted_index]

        return JSONResponse(content={"filename": file.filename, "message": "Prediction successful!", "detail": predicted_class})

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unable to process the image: {str(e)}")
