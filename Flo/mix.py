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
    "Ananassaft":           [0.7, 0.4, 0.1, 0.9, 0.1],
    "Amaretto":             [0.8, 0.1, 0.2, 0.6, 0.2],
    "Anislikör":            [0.6, 0.0, 0.2, 0.4, 0.9],
    "Aperol":               [0.4, 0.1, 0.7, 0.3, 0.1],
    "Apfelsaft":            [0.8, 0.2, 0.0, 0.9, 0.1],
    "Apfelwein":            [0.4, 0.3, 0.1, 0.8, 0.0],
    "Bananenlikör":         [0.9, 0.0, 0.0, 0.7, 0.1],
    "Bananensaft":          [0.9, 0.0, 0.0, 0.7, 0.0],
    "Beerenlikör":          [0.8, 0.3, 0.0, 0.9, 0.2],
    "Bier":                 [0.3, 0.1, 0.2, 0.4, 0.9],
    "Bitter Lemon":         [0.7, 0.3, 0.8, 0.2, 0.1],
    "Birnensaft":           [0.7, 0.1, 0.0, 0.8, 0.1],
    "Blue Curaçao":         [0.7, 0.1, 0.0, 0.8, 0.2],
    "Brauner Zucker":       [1.0, 0.0, 0.1, 0.3, 0.2],
    "Campari":              [0.1, 0.1, 0.9, 0.2, 0.1],
    "Cassislikör":          [0.9, 0.1, 0.0, 1.0, 0.1],
    "Chilisirup":           [0.7, 0.1, 0.0, 0.3, 1.0],
    "Cointreau":            [0.7, 0.1, 0.0, 0.8, 0.2],
    "Cola":                 [0.9, 0.1, 0.1, 0.1, 0.3],
    "Cranberrysaft":        [0.6, 0.9, 0.0, 0.8, 0.1],
    "Dattelsirup":          [1.0, 0.1, 0.0, 0.5, 0.2],
    "Drambuie":             [0.9, 0.1, 0.1, 0.5, 0.4],
    "Eiweiß":               [0.0, 0.0, 0.0, 0.0, 0.2],
    "Erdbeersirup":         [1.0, 0.0, 0.0, 0.9, 0.1],
    "Espresso":             [0.0, 0.1, 0.8, 0.1, 0.7],
    "Fanta":                [0.8, 0.2, 0.0, 0.9, 0.1],
    "Fernet Branca":        [0.1, 0.1, 1.0, 0.2, 0.9],
    "Frangelico":           [0.8, 0.1, 0.1, 0.6, 0.3],
    "Gin":                  [0.2, 0.1, 0.7, 0.2, 0.8],
    "Ginger Ale":           [0.6, 0.2, 0.0, 0.3, 0.7],
    "Ginger Beer":          [0.4, 0.2, 0.1, 0.2, 1.0],
    "Granatapfelsaft":      [0.7, 0.4, 0.1, 0.9, 0.1],   
    "Grapefruitsaft":       [0.5, 0.5, 0.3, 0.7, 0.1],
    "Grenadine":            [0.9, 0.1, 0.1, 0.9, 0.1],
    "Guavensaft":           [0.8, 0.2, 0.0, 1.0, 0.1],
    "Heidelbeerlikör":      [0.8, 0.1, 0.1, 0.9, 0.2],
    "Himbeersirup":         [1.0, 0.1, 0.0, 1.0, 0.1],
    "Holundersaft":         [0.7, 0.2, 0.1, 0.8, 0.2],
    "Holundersirup":        [0.9, 0.1, 0.0, 0.8, 0.0],
    "Ingwer":               [0.0, 0.0, 0.4, 0.2, 1.0],
    "Ingwersirup":          [0.8, 0.2, 0.0, 0.3, 0.7],
    "Jägermeister":         [0.5, 0.1, 0.5, 0.2, 0.8],
    "Kaffee":               [0.1, 0.1, 0.9, 0.1, 0.1],
    "Kaffeelikör":          [0.5, 0.1, 0.8, 0.3, 0.1],
    "Kirschlikör":          [0.8, 0.1, 0.0, 0.9, 0.2],
    "Kirschsaft":           [0.8, 0.5, 0.1, 0.9, 0.0],
    "Kirschwasser":         [0.3, 0.1, 0.1, 0.7, 0.4],
    "Kokosmilch":           [0.4, 0.1, 0.0, 0.5, 0.2],
    "Kokoswasser":          [0.2, 0.1, 0.0, 0.5, 0.0],
    "Kräuterlikör":         [0.6, 0.1, 0.6, 0.2, 0.8],
    "Limettensaft":         [0.2, 0.9, 0.1, 0.6, 0.1],
    "Lycheesaft":           [0.9, 0.2, 0.0, 0.8, 0.1],
    "Mangosaft":            [0.8, 0.2, 0.0, 0.9, 0.1],
    "Maracujasaft":         [0.8, 0.3, 0.0, 1.0, 0.2],
    "Martini":              [0.1, 0.1, 0.9, 0.1, 0.1],
    "Melonensirup":         [1.0, 0.0, 0.0, 0.8, 0.1],
    "Melonenlikör":         [1.0, 0.0, 0.0, 0.9, 0.2],
    "Milch":                [0.4, 0.1, 0.0, 0.0, 0.0],
    "Multivitaminsaft":     [0.9, 0.4, 0.1, 0.9, 0.0],
    "Orangensaft":          [0.8, 0.3, 0.1, 0.9, 0.1],
    "Passionsfruchtsirup":  [1.0, 0.2, 0.0, 1.0, 0.2],
    "Pfirsichlikör":        [0.9, 0.0, 0.0, 0.9, 0.2],
    "Pfirsichsaft":         [0.8, 0.2, 0.0, 0.9, 0.1],
    "Rhabarbersaft":        [0.7, 0.4, 0.0, 0.9, 0.1],
    "Rhabarbersirup":       [0.9, 0.2, 0.0, 0.9, 0.1],
    "Rote-Bete-Saft":       [0.3, 0.2, 0.0, 0.5, 0.7],
    "Rote Saftmischung":    [0.8, 0.3, 0.1, 0.9, 0.1],
    "Rum":                  [0.8, 0.1, 0.2, 0.1, 0.6],
    "Sekt":                 [0.4, 0.2, 0.0, 0.5, 0.3],
    "Sahne":                [0.6, 0.1, 0.0, 0.0, 0.2],
    "Sherry":               [0.4, 0.1, 0.3, 0.4, 0.5],
    "Soda":                 [0.0, 0.1, 0.0, 0.1, 0.1],
    "Sprite":               [0.9, 0.2, 0.1, 0.4, 0.2],
    "Tequila":              [0.2, 0.2, 0.1, 0.1, 0.4],
    "Tonic Water":          [0.6, 0.2, 0.2, 0.3, 0.3],
    "Traubensaft":          [0.9, 0.2, 0.0, 0.8, 0.1],
    "Triple Sec":           [0.6, 0.1, 0.3, 0.4, 0.2],
    "Vanille-Sirup":        [1.0, 0.1, 0.0, 0.3, 0.0],
    "Wein":                 [0.6, 0.3, 0.1, 0.7, 0.2],
    "Wasser":               [0.0, 0.0, 0.0, 0.0, 0.0],
    "Whiskey":              [0.3, 0.1, 0.2, 0.2, 0.5],
    "Wodka":                [0.2, 0.0, 0.2, 0.1, 0.4],
    "Zitronensaft":         [0.5, 0.9, 0.1, 0.6, 0.1],
    "Zucker":               [1.0, 0.0, 0.0, 0.0, 0.0],
    "Zuckersirup":          [1.0, 0.0, 0.0, 0.0, 0.0],
    "Zwetschgenbrand":      [0.5, 0.1, 0.0, 0.6, 0.4],
    "43-er":                [0.8, 0.0, 0.0, 0.3, 0.1],

    "Amarenakirschen":      [0.9, 0.2, 0.0, 0.7, 0.1],
    "Apfelscheiben":        [0.8, 0.4, 0.1, 0.9, 0.0],
    "Basilikum":            [0.0, 0.0, 0.2, 0.2, 0.6],
    "Kakao":                [0.1, 0.0, 0.4, 0.2, 0.0],
    "Minze":                [0.2, 0.1, 0.0, 0.0, 0.7],
    "Muskatnuss":           [0.0, 0.0, 0.5, 0.0, 1.0],
    "Limettenschale":       [0.0, 0.4, 0.7, 0.7, 0.2],
    "Limettenscheiben":     [0.5, 0.9, 0.3, 0.6, 0.1],
    "Orangenschale":        [0.2, 0.2, 0.6, 0.8, 0.2],
    "Orangenscheiben":      [0.8, 0.4, 0.2, 0.9, 0.0],
    "Rosmarin":             [0.1, 0.0, 0.4, 0.1, 0.9],
    "Zimt":                 [0.0, 0.0, 0.4, 0.3, 0.1],
    "Zitronenschale":       [0.0, 0.4, 0.7, 0.7, 0.2],
    "Zitronenscheiben":     [0.5, 0.9, 0.3, 0.6, 0.1],

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
