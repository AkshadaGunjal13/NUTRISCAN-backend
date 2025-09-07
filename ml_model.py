import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

MODEL_PATH = 'model.pkl'

# ---------------------------
# Dataset (mock for demo)
# ---------------------------
data = pd.DataFrame({
    'ingredients': [
        'sugar, milk, cocoa',
        'peanuts, sugar',
        'rice, salt, oil',
        'cheese, eggs, flour',
        'oats, almond milk, honey'
    ],
    'safe': [1, 0, 1, 0, 1]
})

# ---------------------------
# Tokenizer function (no lambda)
# ---------------------------
def ingredient_tokenizer(text):
    return [x.strip() for x in text.split(',')]

# ---------------------------
# Train & save model
# ---------------------------
vectorizer = CountVectorizer(tokenizer=ingredient_tokenizer)
X = vectorizer.fit_transform(data['ingredients'])
y = data['safe']

clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump((vectorizer, clf), f)

# ---------------------------
# Predict safety function
# ---------------------------
def predict_safety(ingredient_text, user_profile=None):
    with open(MODEL_PATH, 'rb') as f:
        vectorizer, clf = pickle.load(f)

    X_test = vectorizer.transform([ingredient_text])
    decision = clf.predict(X_test)[0]

    # Convert to lowercase list
    ingredients_list = [x.strip() for x in ingredient_text.lower().split(',')]

    # Rule-based personalization
    if user_profile:
        # --- Allergies ---
        for allergy in user_profile.get('allergies', []):
            if allergy.lower() in ingredients_list:
                return f"❌ Unsafe: Contains allergen ({allergy})"

        # --- Diet checks ---
        diet = user_profile.get('diet', '').lower()

        if diet == 'vegan':
            non_vegan = ['milk', 'cheese', 'eggs', 'honey', 'butter', 'ghee', 'yogurt']
            for item in non_vegan:
                if item in ingredients_list:
                    return f"❌ Unsafe: Contains {item}, not suitable for Vegan diet"

        elif diet == 'vegetarian':
            non_vegetarian = ['chicken', 'fish', 'meat', 'beef', 'pork', 'mutton', 'shrimp', 'egg', 'gelatin']
            for item in non_vegetarian:
                if item in ingredients_list:
                    return f"❌ Unsafe: Contains {item}, not suitable for Vegetarian diet"

        elif diet == 'keto':
            high_carb = ['sugar', 'rice', 'flour', 'bread', 'potato', 'corn']
            for item in high_carb:
                if item in ingredients_list:
                    return f"❌ Unsafe: Contains {item}, not suitable for Keto diet"

    # If ML model says unsafe (no diet/allergy conflict triggered)
    if decision == 0:
        return "⚠️ Caution: May be unsafe (ML model prediction)"

    return "✅ Safe to eat"
