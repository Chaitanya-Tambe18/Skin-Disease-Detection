# import streamlit as st
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from PIL import Image

# # Load the model
# MODEL_PATH = './model/my_model.h5'
# model = load_model(MODEL_PATH)

# # Define class labels and their disease names
# CLASS_LABELS = ['BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 
#                 'FU-nail-fungus', 'FU-ringworm', 
#                 'PA-cutaneous-larva-migrans', 'VI-vhikenpox', 'VI-shingles']

# DISEASE_NAMES = {
#     'BA-cellulitis': 'Bacterial Infection: Cellulitis',
#     'BA-impetigo': 'Bacterial Infection: Impetigo',
#     'FU-athlete-foot': 'Fungal Infection: Athlete’s Foot',
#     'FU-nail-fungus': 'Fungal Infection: Nail Fungus',
#     'FU-ringworm': 'Fungal Infection: Ringworm',
#     'PA-cutaneous-larva-migrans': 'Parasitic Infection: Cutaneous Larva Migrans',
#     'VI-vhikenpox': 'Viral Infection: Chickenpox',
#     'VI-shingles': 'Viral Infection: Shingles'
# }

# # Streamlit app
# def main():
#     top_image = Image.open('./images/trippyPattern.png')
#     st.sidebar.image(top_image)
    
#     st.sidebar.markdown("> Disclaimer : I do not claim this application as a highly accurate")
#     st.sidebar.markdown("**Note:** You should upload atmost one Image of either class. Since this application is a Classification Task not a Segmentation.")
        


#     st.title("Skin Disease Detection ")
    
#     st.markdown("> An inference API for pre-screening skin diseases based on high-resolution skin lesion images. The API leverages deep learning models to accurately identify and classify various skin conditions, aiding early diagnosis and treatment.")

#     st.write("Upload an image of the affected skin to detect the disease.")

#     # File uploader
#     uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

#     # Prediction button
#     if uploaded_file is not None:
#         # Display uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Predict button
#         if st.button("Predict"):
#             # Preprocess the image for prediction
#             image = np.array(image)
#             image = cv2.resize(image, (224, 224))  # Resize to model input size
#             image = preprocess_input(image)  # Preprocess for the model
#             image = np.expand_dims(image, axis=0)  # Add batch dimension

#             # Make predictions
#             predictions = model.predict(image)
#             predicted_class_index = np.argmax(predictions)
#             predicted_label = CLASS_LABELS[predicted_class_index]

#             # Map predicted label to the exact disease name
#             disease_name = DISEASE_NAMES.get(predicted_label, "Unknown Disease")
            
#             # Display the result
#             st.write(f"### Predicted Disease: **{disease_name}**")
#             st.success("Diagnosis complete!")

# if __name__ == "__main__":
#     main()

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

# Load the model
MODEL_PATH = './model/my_model.h5'
model = load_model(MODEL_PATH)

# Define class labels and their disease names
CLASS_LABELS = ['BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 
                'FU-nail-fungus', 'FU-ringworm', 
                'PA-cutaneous-larva-migrans', 'VI-vhikenpox', 'VI-shingles']

DISEASE_NAMES = {
    'BA-cellulitis': 'Bacterial Infection: Cellulitis',
    'BA-impetigo': 'Bacterial Infection: Impetigo',
    'FU-athlete-foot': 'Fungal Infection: Athlete’s Foot',
    'FU-nail-fungus': 'Fungal Infection: Nail Fungus',
    'FU-ringworm': 'Fungal Infection: Ringworm',
    'PA-cutaneous-larva-migrans': 'Parasitic Infection: Cutaneous Larva Migrans',
    'VI-vhikenpox': 'Viral Infection: Chickenpox',
    'VI-shingles': 'Viral Infection: Shingles'
}

# Streamlit app
def main():
    top_image = Image.open('./images/trippyPattern.png')
    st.sidebar.image(top_image)
    
    st.sidebar.markdown("> Disclaimer : I do not claim this application as a highly accurate")
    st.sidebar.markdown("**Note:** You should upload at most one image of either class. Since this application is a Classification Task, not a Segmentation.")
    
    st.title("Skin Disease Detection")
    
    st.markdown("> An inference API for pre-screening skin diseases based on high-resolution skin lesion images. The API leverages deep learning models to accurately identify and classify various skin conditions, aiding early diagnosis and treatment.")

    st.write("Upload an image of the affected skin to detect the disease.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # Prediction button
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            # Preprocess the image for prediction
            image = np.array(image)
            image = cv2.resize(image, (224, 224))  # Resize to model input size
            image = preprocess_input(image)  # Preprocess for the model
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Make predictions
            predictions = model.predict(image)
            predicted_class_index = np.argmax(predictions)
            confidence = predictions[0][predicted_class_index]
            
            # Set confidence threshold
            CONFIDENCE_THRESHOLD = 0.8  # Adjust as needed
            
            if confidence < CONFIDENCE_THRESHOLD:
                st.warning("The uploaded image does not match any known diseases in the model. Please consult a healthcare professional for further diagnosis.")
            else:
                predicted_label = CLASS_LABELS[predicted_class_index]
                disease_name = DISEASE_NAMES.get(predicted_label, "Unknown Disease")
                st.write(f"### Predicted Disease: **{disease_name}**")
                st.success("Diagnosis complete!")

if __name__ == "__main__":
    main()

