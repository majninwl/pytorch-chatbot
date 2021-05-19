import random
import json


import torch

from model import RNN
from nltk_utils import bag_of_words, tokenize
#from spacy_utils import bag_of_words, return_token
import io

from spellchecker import SpellChecker

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

bot_name = "TANGER MED Bot"
print("Let's chat! (type 'quit' to exit)")



while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)

    spell = SpellChecker(language='fr')

    corrected = []
    for word in sentence:
        corrected.append(spell.correction(word))
        sentence = corrected

    #sentence = return_token(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(-1, 1, 212)
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
        if(tag2 not in greetings):
            break


    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag not in greetings:
                    print(f"{bot_name}: Voulez-vous dire: {intent['patterns'][0]}")
                    response = input("You: ")
                    if response == 'oui':
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                    else:
                        for intent2 in intents['intents']:
                            if tag2 == intent2["tag"]:
                                print(f"{bot_name}: Voulez-vous dire: {intent2['patterns'][0]}")
                                response = input("You: ")
                                if response == 'oui':
                                    print(f"{bot_name}: {random.choice(intent2['responses'])}")
                                else:
                                    print(f"{bot_name}: Veuillez reformuler votre question")
                else:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")



    else:
        print(f"{bot_name}: Je ne comprends pas....")