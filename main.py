from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
from io import BytesIO
from PIL import Image
import json

# initialize Flask application
app = Flask(__name__)

class_names = [
    "n01443537",
    "n01503061",
    "n02084071",
    "n02121808",
    "n02129165",
    "n02274259",
    "n02317335",
    "n02391049",
    "n02510455",
    "n02691156",
    "n02958343",
    "n03991062",
    "n04004767",
    "n04131690",
    "n06874185",
    "n07739125",
]

category_dict = {
    "n02121808": "Domestic Cat",
    "n02084071": " Dog",
    "n01503061": " Bird",
    "n02510455": " Giant Panda",
    "n01443537": " Gold Fish",
    "n02958343": " Car",
    "n02317335": " Starfish",
    "n07739125": " Apple",
    "n02391049": " Zebra",
    "n02691156": " Airplane",
    "n03991062": " Flower Pot",
    "n02274259": " Butterfly",
    "n02129165": " Lion",
    "n04131690": " Salt or Pepper Shaker",
    "n04004767": " Printer",
    "n06874185": " Traffic Light",
}
model_path = "./models/"
model_names = ["RapidNet", "VGG16", "VGG19", "InceptionV3", "DenseNet201"]
model_files = [
    "RapidNet.h5",
    "VGG16.h5",
    "VGG19.h5",
    "InceptionV3.h5",
    "DenseNet201.h5",
]


@app.route("/predict-all", methods=["POST"])
def predictAll():
    """
    Predict the image class
    """

    file = request.files["image"]

    img = Image.open(BytesIO(file.read()))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0  # normalize pixel values
    img_array = img_array.reshape((1,) + img_array.shape)  # add batch dimension

    predictionData = []

    for i in range(len(model_names)):
        # # read the image file and convert it to a numpy array
        modelName = model_names[i]
        model = load_model(model_path + model_files[i])
        prediction = model.predict(img_array)

        class_index = np.argmax(prediction)
        predictionData.append(
            {
                "model": modelName,
                "category": category_dict[class_names[class_index]],
                "probability": str(prediction[0][class_index]),
            }
        )

    print("predictionData for current request: ", predictionData)

    return jsonify({"data": predictionData})

@app.route("/verify", methods=["POST"])
def verify():
    """
    Verify the image class
    """

    file = request.files["image"]

    img = Image.open(BytesIO(file.read()))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape((1,) + img_array.shape)

    predictionData = []

    predictionCount = {}

    for i in range(len(model_names)):
        # # read the image file and convert it to a numpy array
        modelName = model_names[i]
        model = load_model(model_path + model_files[i])
        prediction = model.predict(img_array)

        class_index = np.argmax(prediction)
        predictionData.append(
            {
                "model": modelName,
                "category": class_names[class_index],
                "probability": str(prediction[0][class_index]),
            }
        )
        predictionCount[class_names[class_index]] = predictionCount.get(
            class_names[class_index], 0
        ) + 1

    print("predictionData for current upload request: ", predictionData)

    verifiedCategory = max(predictionCount, key=predictionCount.get)

    return jsonify({"data": {'verifiedCategory': verifiedCategory}})


if __name__ == "__main__":
    app.run(debug=True)
