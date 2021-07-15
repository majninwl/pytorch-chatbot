from flask import Flask, jsonify, request, session, render_template
from flask_cors import CORS
import io
import json
import random
import chat
import os

app = Flask(__name__)
app.static_folder = 'static'

app.secret_key = 'adfnSjgnbG375y36y96tuwlgjslh'
CORS(app, supports_credentials=True)

first_tag = ""
second_tag = ""
first_sug_is_correct = True

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def chat_response():  

    global first_tag
    global second_tag
    global first_sug_is_correct
        
    user_input = request.form['user_input']
    
    if user_input not in ['oui', 'non']:

        first_tag, second_tag, first_suggestion = chat.chatbot(user_input)
        #session['first_tag'], session['second_tag'], first_suggestion = chat.chatbot(user_input)
        
        #session['first_sug_is_correct'] = True
        first_sug_is_correct = True

        return jsonify({'response': first_suggestion})
        
    else:
        
        first_response = ""
        second_response = ""
        second_suggestion = ""
        
        with io.open('intents.json', 'r', encoding='utf8') as json_data:
            intents = json.load(json_data)

        #first_tag = session.get('first_tag')
        #second_tag = session.get('second_tag')
        #first_sug_is_correct = session.get('first_sug_is_correct')


        for intent in intents['intents']:
            if first_tag == intent["tag"]:
                    first_response = str(f"{random.choice(intent['responses'])}")
            if second_tag == intent["tag"]:
                    second_suggestion = str(f"Voulez-vous dire: {intent['patterns'][0]}")
                    second_response = str(f"{random.choice(intent['responses'])}")
        

        yes_or_no = user_input
        
        if yes_or_no == 'oui' and first_sug_is_correct:
            first_sug_is_correct = True
            return jsonify({'response': first_response})

        if yes_or_no == 'non' and first_sug_is_correct:
            first_sug_is_correct = False
            return jsonify({'response': second_suggestion})
        
        if yes_or_no == 'oui' and not first_sug_is_correct:
            first_sug_is_correct = True
            return jsonify({'response': second_response})

        if yes_or_no == 'non' and not first_sug_is_correct:
            first_sug_is_correct = True
            return jsonify({'response': 'Veuillez reformuler votre question'})


def check_ping():
    hostname = "127.0.0.1:5500"
    response = os.system("ping -c 1 " + hostname)
    # and then check the response...
    if response == 0:
        pingstatus = "Network Active"
    else:
        pingstatus = "Network Error"

    return pingstatus

if __name__ == '__main__':
    app.run(debug=True)