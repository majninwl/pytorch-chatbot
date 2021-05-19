from flask import Flask, jsonify, request, session
import io
import json
import random
import chat

app = Flask(__name__)
app.secret_key = 'abc'


@app.route("/", methods=['GET'])
def chat_response():
    user_input = request.json['user_input']

    if (user_input not in ['oui', 'non']):
        list = chat.chatbot(user_input)

        session['first_tag'] = str(list[0])
        session['second_tag'] = str(list[1])

        first_suggestion = str(list[2])

        session['first_sug_is_correct'] = True

        return jsonify({'response': first_suggestion})

    else:

        first_response = ""
        second_response = ""
        second_suggestion = ""

        with io.open('intents.json', 'r', encoding='utf8') as json_data:
            intents = json.load(json_data)

        for intent in intents['intents']:
            if session['first_tag'] == intent["tag"]:
                first_response = str(f"{random.choice(intent['responses'])}")
            if session['second_tag'] == intent["tag"]:
                second_suggestion = str(f"Voulez-vous dire: {intent['patterns'][0]}")
                second_response = str(f"{random.choice(intent['responses'])}")

        yes_or_no = request.json['user_input']

        if yes_or_no == 'oui' and session['first_sug_is_correct']:
            session['first_sug_is_correct'] = True
            return jsonify({'response': first_response})

        if yes_or_no == 'non' and session['first_sug_is_correct']:
            session['first_sug_is_correct'] = False
            return jsonify({'response': second_suggestion})

        if yes_or_no == 'oui' and not session['first_sug_is_correct']:
            session['first_sug_is_correct'] = True
            return jsonify({'response': second_response})

        if yes_or_no == 'non' and not session['first_sug_is_correct']:
            session['first_sug_is_correct'] = True
            return jsonify({'response': 'Veuillez reformuler votre question'})


if __name__ == '__main__':
    app.run(debug=True)