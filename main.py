import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64
from groq import Groq
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
# ---------- Load Model and Class Dictionary ----------
model = load_model("artifacts/transfer_learning.h5")
class_dict = np.load("artifacts/class_names.npy", allow_pickle=True)

# ---------- Benefits Dictionary ----------
plant_benefits = {
    'Alpinia Galanga (Rasna)': "Used for pain relief and digestive issues.",
    'Amaranthus Viridis (Arive-Dantu)': "Good for anemia and inflammation.",
    'Artocarpus Heterophyllus (Jackfruit)': "Rich in nutrients, aids digestion.",
    'Azadirachta Indica (Neem)': "Purifies blood, treats skin disorders.",
    'Basella Alba (Basale)': "Rich in iron and vitamins.",
    'Brassica Juncea (Indian Mustard)': "Used for joint pain and respiratory issues.",
    'Carissa Carandas (Karanda)': "Boosts digestion and treats anemia.",
    'Citrus Limon (Lemon)': "Rich in Vitamin C, aids immunity.",
    'Ficus Auriculata (Roxburgh fig)': "Used for diabetes management.",
    'Ficus Religiosa (Peepal Tree)': "Used for asthma and diabetes.",
    'Hibiscus Rosa-sinensis': "Good for hair health and digestive issues.",
    'Jasminum (Jasmine)': "Used for stress relief and skin care.",
    'Mangifera Indica (Mango)': "Rich in antioxidants, boosts immunity.",
    'Mentha (Mint)': "Aids digestion and relieves headaches.",
    'Moringa Oleifera (Drumstick)': "Rich in vitamins, anti-inflammatory.",
    'Muntingia Calabura (Jamaica Cherry-Gasagase)': "Used for pain relief.",
    'Murraya Koenigii (Curry)': "Good for hair and digestion.",
    'Nerium Oleander (Oleander)': "Used externally for skin issues.",
    'Nyctanthes Arbor-tristis (Parijata)': "Used for fever and joint pain.",
    'Ocimum Tenuiflorum (Tulsi)': "Boosts immunity, good for respiratory health.",
    'Piper Betle (Betel)': "Used for digestive health.",
    'Plectranthus Amboinicus (Mexican Mint)': "Good for cough and cold.",
    'Pongamia Pinnata (Indian Beech)': "Used for skin diseases and wounds.",
    'Psidium Guajava (Guava)': "Rich in Vitamin C, boosts immunity.",
    'Punica Granatum (Pomegranate)': "Rich in antioxidants, good for blood health.",
    'Santalum Album (Sandalwood)': "Used for skin care and cooling.",
    'Syzygium Cumini (Jamun)': "Used for diabetes management.",
    'Syzygium Jambos (Rose Apple)': "Good for digestive health.",
    'Tabernaemontana Divaricata (Crape Jasmine)': "Used for pain relief.",
    'Trigonella Foenum-graecum (Fenugreek)': "Good for diabetes and digestion."
}

# ---------- Utility Functions ----------
def predict(image):
    IMG_SIZE = (1, 224, 224, 3)
    img = image.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)
    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)
    return pred

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- Streamlit App ----------
if __name__ == '__main__':
    add_bg_from_local("artifacts/Background.jpg")
    st.markdown(
        '<p style="font-family:sans-serif; color:White; font-size: 42px;">Medicinal Leaf Classification</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:White;'>Upload a leaf image to classify and ask detailed questions about its medicinal properties.</p>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload a leaf image")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.resize((300, 300))
        st.image(img)

        if st.button("Predict"):
            pred = predict(img)
            name = class_dict[pred]
            st.session_state['predicted_leaf'] = name

            # Retrieve benefits
            benefits = plant_benefits.get(name, "Benefits information not available.")

            st.markdown(
                f'<p style="font-family:sans-serif; color:White; font-size: 20px;">🌿 <b>{name}</b></p>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<p style="font-family:sans-serif; color:White; font-size: 16px;">🩺 Benefits: {benefits}</p>',
                unsafe_allow_html=True
            )

    if 'predicted_leaf' in st.session_state:
        st.markdown("<h4 style='color:white;'>💬 Ask questions about this leaf:</h4>", unsafe_allow_html=True)
        user_question = st.chat_input("Ask your question here...")

        if user_question:
            with st.chat_message("user"):
                st.write(user_question)

            leaf_name = st.session_state['predicted_leaf']
            prompt = (
                f"You are a medicinal plants expert. Provide a clear, student-friendly, detailed answer "
                f"in less than 150 words about the following identified plant:\n\n"
                f"Plant: {leaf_name}\n\n"
                f"User's Question: {user_question}\n\n"
                f"Explain in structured bullet points if possible."
            )

            try:
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",   # ✅ Updated, active model
                    messages=[
                        {"role": "system", "content": "You are a helpful medicinal plant assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = completion.choices[0].message.content
            except Exception as e:
                answer = f"An error occurred while fetching data from Groq: {e}"

            with st.chat_message("assistant"):
                st.markdown(
                    f"<div style='color: white; font-size: 16px;'>{answer}</div>",
                    unsafe_allow_html=True
    )
