from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

# Load the model and scaler from disk
with open('knn_model.pkl', 'rb') as model_file:
    knn_loaded = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler_loaded = pickle.load(scaler_file)

# Load the comprehensive nutritional dataset
nutritional_data = pd.read_excel('NutritionalFacts.xlsx')


class UserProfile(BaseModel):
    age: int
    gender: str
    favorite_fruits: list
    disliked_fruits: list
    fruit_allergies: str
    fruit_intolerances: str
    health_objectives: str
    activity_level: str
    meals_per_day: int


def get_preferred_foods(user_profile, nutritional_data):
    if user_profile.fruit_allergies != "None":
        pass  # Apply allergy filters here
    if user_profile.fruit_intolerances != "None":
        pass  # Apply intolerance filters here

    # Ensure column name is correct
    if 'Name' not in nutritional_data.columns:
        raise ValueError("The 'Name' column is not present in the dataset")

    # Filter out disliked fruits
    preferred_foods = nutritional_data[~nutritional_data['Name'].isin(user_profile.disliked_fruits)]
    return preferred_foods



def recommend_foods(user_profile, nutritional_data, knn_model, scaler):
    preferred_foods = get_preferred_foods(user_profile, nutritional_data)
    preferred_features = preferred_foods.drop(columns=['Name', 'Category'])

    if preferred_features.empty:
        return []

    scaled_preferred_features = scaler.transform(preferred_features)
    query_point = np.mean(scaled_preferred_features, axis=0).reshape(1, -1)

    distances, indices = knn_model.kneighbors(query_point)
    recommended_foods = preferred_foods.iloc[indices[0]]

    return recommended_foods


@app.post("/recommendations")
def get_recommendations(user_profile: UserProfile):
    recommendations = recommend_foods(user_profile, nutritional_data, knn_loaded, scaler_loaded)
    if recommendations.empty:
        raise HTTPException(status_code=404, detail="No recommendations found")
    return recommendations.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
