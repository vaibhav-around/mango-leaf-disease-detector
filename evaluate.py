import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter


from PIL import Image
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix





# Configurations

datasetDir = "./evaluation-dataset"
failuresDir = "faliures"
reportsDir = "reports"


verifierThreshhold = 0.85
diseaseConfidenceThreshold = 60


os.makedirs(failuresDir, exist_ok=True)
os.makedirs(reportsDir, exist_ok=True)


# Loading models

verifier_model = tf.keras.models.load_model(
    "mango_leaf_verification.keras"
)


disease_model = tf.keras.models.load_model(
    "mango_leaf_disease_model_v2.keras"
)

# classes


class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
    "Unknown"
]


# preprocess


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224,224))
    img_array = np.array(image)

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)

    verifier_score = verifier_model.predict(
        img_array,
        verbose = 0
    )[0][0]

    prediction = disease_model.predict(
        img_array,
        verbose = 0
    )[0]

    confidence = np.max(prediction) * 100

    if confidence < diseaseConfidenceThreshold:
        return "Unknown", confidence


    predicted_index = np.argmax(prediction)

    return class_names[predicted_index], confidence


#Evaluation 


results = []

y_true = []

y_pred = []
healthy_correct = 0
healthy_total = 0

diseased_correct = 0
diseased_total = 0

unknown_correct = 0
unknown_total = 0
healthy_prediction_counter = Counter()

for actual_class in os.listdir(datasetDir):
    class_dir = os.path.join(
        datasetDir,
        actual_class
    )


    if not os.path.isdir(class_dir):
        continue

    for image_name in os.listdir(class_dir):

        image_path = os.path.join(
            class_dir,
            image_name
        )
        predicted_class, confidence = predict_image(
            image_path
        )


        
       
        if actual_class == "healthy_mango":
            healthy_prediction_counter[predicted_class] += 1;
            healthy_total += 1
        
            if predicted_class == "Healthy":
                healthy_correct += 1


        elif actual_class == "diseased_mango":

            diseased_total += 1

            if predicted_class not in [
                "Healthy",
                    "Unknown"
            ]:
                diseased_correct += 1


        elif actual_class in [
            "animals",
            "people",
            "objects",
            "scenes",
            "other_leaves"
            ]:

                unknown_total += 1

                if predicted_class == "Unknown":
                    unknown_correct += 1


        results.append(
            {
                "image":image_name,
                "actual": actual_class,
                "predicted": predicted_class,
                "confidence": confidence
            }
        )

        if predicted_class != actual_class:
            fail_dir = os.path.join(
                failuresDir,
                actual_class
            )

            os.makedirs(fail_dir, exist_ok=True)

            shutil.copy(
                image_path,
                os.path.join(
                    fail_dir,
                    image_name
                )
            )


# --------------------
# SAVE REPORTS
# --------------------

df = pd.DataFrame(results)

df.to_csv(
    os.path.join(
        reportsDir,
        "evaluation_report.csv"
    ),
    index=False
)

print()

print("healthy_correct =", healthy_correct)
print("healthy_total   =", healthy_total)

healthy_accuracy = (
    healthy_correct / healthy_total * 100
)


print("diseased_correct =", diseased_correct)
print("diseased_total   =", diseased_total)
diseased_accuracy = (
    diseased_correct / diseased_total * 100
)

print("unknown_correct =", unknown_correct)
print("unknown_total   =", unknown_total)
unknown_accuracy = (
    unknown_correct / unknown_total * 100
)

overall_accuracy = (
    (
        healthy_correct +
        diseased_correct +
        unknown_correct
    )
    /
    (
        healthy_total +
        diseased_total +
        unknown_total
    )
) * 100


print("\n========== HEALTHY IMAGE ANALYSIS ==========\n")

for disease, count in sorted(
    healthy_prediction_counter.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{disease}: {count}")

print("\n========== RESULTS ==========\n")

print(
    f"Healthy Accuracy: "
    f"{healthy_accuracy:.2f}%"
)

print(
    f"Diseased Accuracy: "
    f"{diseased_accuracy:.2f}%"
)

print(
    f"Unknown Detection Accuracy: "
    f"{unknown_accuracy:.2f}%"
)

print(
    f"Overall Accuracy: "
    f"{overall_accuracy:.2f}%"
)