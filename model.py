from sentence_transformers import SentenceTransformer

# Choose a proper directory
model_save_path = "C:/Users/Sameh/Desktop/Omar Dev/Fastapi_Service/model/paraphrase-MiniLM-L6-v2"

# Download the model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Save the model to your directory
model.save(model_save_path)

print(f"Model downloaded and saved to {model_save_path}")
