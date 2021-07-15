import random
import json


import torch

from model import RNN
from nltk_utils import bag_of_words, tokenize
#from spacy_utils import bag_of_words, return_token
import io

from spellchecker import SpellChecker


directly_answered = ['Salutations', 'au revoir', 'merci', 'contact']

def chatbot(user_input):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with io.open('intents.json', 'r', encoding='utf8') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]
    num_classes = data["num_classes"]
    num_layers = data["num_layers"]


    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(model_state)
    model.eval()

    sentence = user_input
    sentence = tokenize(sentence)

    non_spellchecked = ['colisage','DS', 'AMPE', 'AMPI', 'AMP','Morocco', 'Foodex', 'DUM', 'BAD', 'CP']

    spell = SpellChecker(language='fr')

    corrected = []
    for word in sentence:
        if word in non_spellchecked or word.lower() in non_spellchecked:
            corrected.append(word)
        else:
            corrected.append(spell.correction(word))
    sentence = corrected

    #sentence = return_token(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(-1, 1, 210)
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)
    _, predictions = torch.sort(output, dim=1, descending=True)

    greetings = ['Salutations', 'au revoir', 'merci']

    tag = tags[predicted.item()]

    i=1

    while True:
        tag2 = tags[predictions[0].numpy()[i]]
        i = i+1
        if(tag2 not in directly_answered):
            break

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if user_input == "/start":
                    chatbot_response = "Bonjour, je suis votre assistant Tanger Med Bot 😄. Il y a beaucoup de questions aux quelles je peux répondre."
                else:
                    if tag not in directly_answered:
                        chatbot_response = f"Voulez-vous dire: {intent['patterns'][0]}"
                    else:
                        chatbot_response = f"{random.choice(intent['responses'])}"
    else:
        chatbot_response = "Je ne comprends pas...."

    return tag, tag2, chatbot_response