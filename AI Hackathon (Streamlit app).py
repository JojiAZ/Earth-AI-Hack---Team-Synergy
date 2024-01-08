import streamlit as st
import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration


# Load fine-tuned BART model for industry prediction
industry_model_path = r"C:\Users\anton\OneDrive\Desktop\Hackathon\bart_finetuned"
industry_tokenizer = BartTokenizer.from_pretrained(industry_model_path, local_files_only=True)
industry_model = BartForConditionalGeneration.from_pretrained(industry_model_path)

def generate_prediction(prompt):
    # Tokenize and generate industry prediction using fine-tuned BART model
    industry_inputs = industry_tokenizer(prompt, return_tensors='pt', max_length=64, truncation=True)
    industry_outputs = industry_model.generate(**industry_inputs, max_length=16, num_beams=5, length_penalty=0.6)
    industry_prediction = industry_tokenizer.decode(industry_outputs[0], skip_special_tokens=True)

    return industry_prediction

def main():
    st.title("Circularity Matrix Quadrants and Text Summarization")

    # Layout for the left side (Gartner-style quadrant chart)
    st.sidebar.header("Circularity Matrix Quadrants")

    # Define information for each quadrant
    quadrants = [
        {
            "title": "Hard to Access and Hard to Process",
            "description": "Precludes easy repair and remanufacturing (e.g., Tire). Loss of the value embedded in them.",
            "x": "Hard",
            "y": "Hard",
            "color": "red"
        },
        {
            "title": "Easy to Access but Hard to Process",
            "description": "Relatively low-embedded-value products such as carpets, mattresses, and athletic footwear. Products can be recouped from the consumer, but they can't be easily reconditioned, and extracting materials from them is complex.",
            "x": "Hard",
            "y": "Easy",
            "color": "yellow"
        },
        {
            "title": "Hard to Access but Easy to Process",
            "description": "Products whose use makes them difficult to retrieve. Takeout food packaging, for example, may contain easily recyclable materials but often winds up in landfills because of the food residue on it, which is costly to remove.",
            "x": "Easy",
            "y": "Hard",
            "color": "blue"
        },
        {
            "title": "Easy to Access and Easy to Process",
            "description": "Components for which a well-oiled recycling infrastructure already exists. Here there is plenty of scope for companies not already in the circular economy.",
            "x": "Easy",
            "y": "Easy",
            "color": "green"
        }
    ]

    # Create a DataFrame for plotting
    df = pd.DataFrame(quadrants)

    # Plot the quadrants
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["x"], df["y"], c=df["color"], alpha=0.7, edgecolors="k", linewidths=1)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_title("Circularity Matrix Quadrants")
    ax.set_xlabel("Ease of Access")
    ax.set_ylabel("Ease of Processing")
    ax.grid(True)
    plt.tight_layout()

    # Display the plot in the sidebar
    st.sidebar.pyplot(fig)

    # Layout for the right side (Text summarization output)
    st.header("Text Summarization Output")

    # Enlarge the sidebar
    st.markdown('<style>div.Widget.row-widget.stRadio > div{flex-direction:column;}</style>', unsafe_allow_html=True)

    # Bigger text area for user prompt
    user_prompt = st.text_area("User Prompt:", "", height=200)

    if st.button("Generate Prediction"):
        if user_prompt:
            # Generate industry prediction
            industry = generate_prediction(user_prompt)

            # Display user prompt and industry prediction
            st.text(f"User Prompt: {user_prompt}")
            st.text(f"Industry Prediction: {industry}")

if __name__ == "__main__":
    main()



