# Konzeptidee zur Projektarbeit: 

# Eine sehr einfache Art der "unsupervised learning" ist die Verwendung von Autoencodern. Vereinfach gesagt, habe ich versucht ein Modell zu erstellen, das die Geschmacksprofile von Zutaten aus einem Datensatz lernt (die Geschmäcker sind geschätzt, ich habe keine Ahnung wie Rum, Gin, etc. wirklich schmecken) und dann basierend auf einem vom Benutzer eingegebenen Geschmacksprofil einen passenden Cocktail vorschlägt.

# Das Modell ist ein einfacher Autoencoder mit einer Input-Schicht, einer versteckten Schicht und einer Output-Schicht. Die Input-Schicht hat die gleiche Anzahl von Neuronen wie die Anzahl der Geschmacksprofile (süß, sauer, bitter, fruchtig, würzig), die versteckte Schicht hat nur 3 Neuronen, um die Daten zu komprimieren, und die Output-Schicht hat wieder die gleiche Anzahl von Neuronen wie die Input-Schicht.

# Also es ist quasi ein Karteikarten-Prinzip. Der Encoder komprimiert die Informationen auf 3 Karteikarten und der Decoder entpackt sie wieder und versucht daraus den Cocktail zu "erraten".

# Man gibt einfach sein Geschmacksprofil ein und das Modell gibt einem ein Getränk zurück, wekches bestmöglich zu dem Profil passt. Aufgrund der geringen Anzahl von Zutaten und Geschmacksprofilen ist das Modell sehr einfach und nicht sehr genau, aber es zeigt das Konzept eines Autoencoders. 

# Cooler wäre natürlich ein Reinforcement Learning Ansatz, bei dem das Modell tatsächlich neue Mixe generiert, evtl. noch mit einer Belohnungsfunktion, aber ich habe noch nicht so ganz verstanden, wie das funktioniert. :D 


# Author: Florian Grassl
# Date: 11.11.2024





import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np


# Geschmacksprofile der Zutaten (süß, sauer, bitter, fruchtig, würzig) - Werte zwischen 0 und 1 (geschätzt) 

ingredients = {
    "Rum": [0.8, 0.1, 0.2, 0.1, 0.6],
    "Gin": [0.2, 0.1, 0.7, 0.2, 0.8],
    "Wodka": [0.1, 0.0, 0.1, 0.1, 0.1],
    "Zitronensaft": [0.1, 0.9, 0.1, 0.1, 0.1],
    "Orangensaft": [0.7, 0.3, 0.1, 0.9, 0.1],
    "Zucker": [1.0, 0.0, 0.0, 0.0, 0.0],
    "Minze": [0.2, 0.1, 0.0, 0.0, 0.7],
    "Triple Sec": [0.6, 0.1, 0.3, 0.4, 0.2],
    "Cola": [0.9, 0.1, 0.1, 0.1, 0.3],
    "Tonic Water": [0.1, 0.2, 0.1, 0.1, 0.1],
    "Grenadine": [0.9, 0.1, 0.1, 0.9, 0.1],
    "Ananassaft": [0.8, 0.1, 0.1, 0.9, 0.1],
    "Soda": [0.1, 0.1, 0.1, 0.1, 0.1],
    "Wermut": [0.1, 0.1, 0.9, 0.1, 0.1],
    "Kaffee": [0.1, 0.1, 0.9, 0.1, 0.1],
}

ingredient_names = list(ingredients.keys())
ingredient_profiles = np.array(list(ingredients.values()))

# Autoencoder-Architektur
input_dim = ingredient_profiles.shape[1]

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(3, activation='relu')(input_layer)  # Komprimiert auf 3 Dimensionen

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder-Modell
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile Modell
autoencoder.compile(optimizer='adam', loss='mse')

# Trainiere das Modell auf die Zutaten-Geschmacksprofile
autoencoder.fit(ingredient_profiles, ingredient_profiles, epochs=1000, batch_size=4, verbose=0)

# Funktion zur Umwandlung einer Geschmacksanfrage in ein Profil
def parse_taste_input(taste_input):
    taste_dict = {"süß": 0, "sauer": 1, "bitter": 2, "fruchtig": 3, "würzig": 4}
    user_profile = [0.0] * len(taste_dict)
    
    # Aufteilen des Eingabe-Strings und Zuweisung der Werte
    for part in taste_input.split(","):
        value, taste = part.strip().split()
        value = float(value)
        taste = taste.lower()
        
        if taste in taste_dict:
            user_profile[taste_dict[taste]] = value
    
    return user_profile

def generate_cocktail(user_profile):
    compressed_profile = autoencoder.predict(np.array([user_profile]))[0]
    closest_idx = np.argmin(np.sum(np.square(ingredient_profiles - compressed_profile), axis=1))
    return ingredient_names[closest_idx]

# Beispielablauf
taste_input = input("Gib deinen gewünschten Geschmack ein (z.B. '0.4 süß, 0.8 sauer'): ")
user_profile = parse_taste_input(taste_input)

generated_cocktail = generate_cocktail(user_profile)
print("Empfohlener Drink für den gewünschten Geschmack:", generated_cocktail)
