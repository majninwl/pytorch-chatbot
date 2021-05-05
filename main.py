from flask import Flask, jsonify, request
import io
import json
import random
import chat

app = Flask(__name__)

first_tag = ""
second_tag = ""
bot_name = "TANGER MED Bot"
first_sug_is_correct = True

@app.route("/", methods=['GET'])
def index():
    global first_tag
    global second_tag
    
    user_input = request.json['user_input']
    list = chat.chatbot(user_input)
    first_tag = str(list[0])
    second_tag = str(list[1])
    first_suggestion = str(list[2])

    return jsonify({'response': first_suggestion})


@app.route("/", methods=['POST'])
def chat_response():
    global first_sug_is_correct

    with io.open('intents.json', 'r', encoding='utf8') as json_data:
        intents = json.load(json_data)

    response = ""
    second_suggestion = ""

    yes_or_no = request.json['user_input']
    
    if yes_or_no == 'oui' and first_sug_is_correct:
        for intent in intents['intents']:
            if first_tag == intent["tag"]:
                response = str(f"{bot_name}: {random.choice(intent['responses'])}")
                return jsonify({'response': response})

    if yes_or_no == 'non' and first_sug_is_correct:
        first_sug_is_correct = False
        for intent in intents['intents']:
            if second_tag == intent["tag"]:
                second_suggestion = str(f"{bot_name}: Voulez-vous dire: {intent['patterns'][0]}")
                return jsonify({'response': second_suggestion})
    
    if yes_or_no == 'oui' and not first_sug_is_correct:
        for intent in intents['intents']:
            if second_tag == intent["tag"]:
                response = str(f"{bot_name}: {random.choice(intent['responses'])}")
                return jsonify({'response': response})

    if yes_or_no == 'non' and not first_sug_is_correct:
        first_sug_is_correct = True
        return jsonify({'response': 'Veuillez reformuler votre question'})

    return jsonify({'response': 'Je ne comprends pas'})




if __name__ == '__main__':
    app.run(debug=True)