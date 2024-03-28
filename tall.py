import pickle

# Charger le modèle depuis le fichier
model = pickle.load(open('c:/Users/talla/Music/Memoire/memoireApp/memoireknn_model.pkl', 'rb'))

# Vérifier le type de l'objet
print(type(model))
