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

    # Layout for the left side (Gartner-style quadrant chart)
    st.sidebar.header("EcoVision.AI: Circular Business Evaluator")

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
            "x": "Easy",
            "y": "Hard",
            "color": "yellow"
        },
        {
            "title": "Hard to Access but Easy to Process",
            "description": "Products whose use makes them difficult to retrieve. Takeout food packaging, for example, may contain easily recyclable materials but often winds up in landfills because of the food residue on it, which is costly to remove.",
            "x": "Hard",
            "y": "Easy",
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

    # Dummy data for the number of ideas in each quadrant
    data_points = [
        {"quadrant": "Easy to access and easy to process", "ideas": 4},
        {"quadrant": "Hard to access but easy to process", "ideas": 2},
        {"quadrant": "Easy to access but hard to process", "ideas": 1},
        {"quadrant": "Hard to access and hard to process", "ideas": 3},
    ]

    # Create a DataFrame for the additional data points
    scatter_df = pd.DataFrame(data_points)

    # Quadrant mapping for scatter plot
    quadrant_mapping = {
        "Easy to access and easy to process": {"x": 0.5, "y": -0.5},
        "Hard to access but easy to process": {"x": -0.5, "y": -0.5},
        "Easy to access but hard to process": {"x": 0.5, "y": 0.5},
        "Hard to access and hard to process": {"x": -0.5, "y": 0.5},
    }

    # Plot the quadrants
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["x"], df["y"], c=df["color"], alpha=0.7, edgecolors="k", linewidths=1)

    # Scatter plot additional data points
    for index, row in scatter_df.iterrows():
        x, y = quadrant_mapping[row["quadrant"]]["x"], quadrant_mapping[row["quadrant"]]["y"]
        ax.text(x, y, str(row["ideas"]), ha="center", va="center", fontsize=20, fontweight="bold", color="black")

        # Adjust the text coordinates to the middle of each quadrant
        middle_x = (x + quadrant_mapping[row["quadrant"]]["x"]) / 2
        middle_y = (y + quadrant_mapping[row["quadrant"]]["y"]) / 2

        # Display the data points in the middle of each quadrant
        ax.text(middle_x, middle_y, str(row["ideas"]), ha="center", va="center", fontsize=20, fontweight="bold", color="black")

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_title("Mapping business ideas", fontsize=18)
    ax.set_xlabel("Ease of Access", fontsize=14)
    ax.set_ylabel("Ease of Processing", fontsize=14)
    ax.set_xticks([-0.5, 0.5])
    ax.set_yticks([-0.5, 0.5])
    ax.set_xticklabels(["Easy", "Hard"])
    ax.set_yticklabels(["Easy", "Hard"])
    ax.grid(True)
    plt.tight_layout()

    # Display the plot in the sidebar
    st.sidebar.pyplot(fig)

    # Layout for the right side (Text summarization output)
    st.header(" Enter Your Business Concept for Circular Economy Evaluation")

    # Enlarge the sidebar
    st.markdown('<style>div.Widget.row-widget.stRadio > div{flex-direction:column;}</style>', unsafe_allow_html=True)

    # Bigger text area for user prompt
    user_prompt = st.text_area("", "", height=200)

    if st.button("Generate Prediction"):
        if user_prompt:
            # Generate industry prediction
            industry = generate_prediction(user_prompt)

            # Dummy values for Industry, Business Model, and Quadrant
            industry_dummy = "Fashion, Technology"
            _dummy = "Design for Recycling (DFR)"
            quadrant_dummy = "Easy to access and easy to process"

            # Display user prompt, industry prediction, and dummy values
            st.text(f"Industry: {industry_dummy}")
            st.text(f"Business Model: {_dummy}")
            st.text(f"Quadrant: {quadrant_dummy}")

if __name__ == "__main__":
    main()
