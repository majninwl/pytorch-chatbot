# from telegram import telegram_chatbot
import io
import json
import random
import chat
from flask import Flask, request, Response, session
import requests
import configparser as cfg

app = Flask(__name__)
app.secret_key = 'abc'
# bot = telegram_chatbot("config.cfg")


first_tag = ""
second_tag = ""
first_sug_is_correct = True


def make_reply(msg):
    global first_tag
    global second_tag
    global first_sug_is_correct

    if (msg not in ['oui', 'non']):

        first_tag, second_tag, first_suggestion = chat.chatbot(msg)
        # session['first_tag'], session['second_tag'], first_suggestion = chat.chatbot(msg)

        # first_tag = session['first_tag']
        # second_tag = session['second_tag']

        # session['first_sug_is_correct'] = True
        first_sug_is_correct = True

        return first_suggestion

    else:

        first_response = ""
        second_response = ""
        second_suggestion = ""

        with io.open('intents.json', 'r', encoding='utf8') as json_data:
            intents = json.load(json_data)

        for intent in intents['intents']:
            if first_tag == intent["tag"]:
                first_response = str(f"{random.choice(intent['responses'])}")
            if second_tag == intent["tag"]:
                second_suggestion = str(f"Voulez-vous dire: {intent['patterns'][0]}")
                second_response = str(f"{random.choice(intent['responses'])}")

        yes_or_no = msg

        if yes_or_no == 'oui' and first_sug_is_correct:
            first_sug_is_correct = True
            return first_response

        if yes_or_no == 'non' and first_sug_is_correct:
            first_sug_is_correct = False
            return second_suggestion

        if yes_or_no == 'oui' and not first_sug_is_correct:
            first_sug_is_correct = True
            return second_response

        if yes_or_no == 'non' and not first_sug_is_correct:
            first_sug_is_correct = True
            return 'Veuillez reformuler votre question'


def read_token_from_config_file(config):
    parser = cfg.ConfigParser()
    parser.read(config)
    return parser.get('creds', 'token')


def send_message(chat_id, text):
    token = read_token_from_config_file("config.cfg")

    url = f'https://api.telegram.org/bot{token}/sendMessage'

    payload = {'chat_id': chat_id, 'text': text}

    yes_no_keyboard = {'reply_markup': {'keyboard':
                                            [[{"text": "oui"}],
                                             [{"text": "non"}]],
                                        "resize_keyboard": True,
                                        "one_time_keyboard": True}}

    if text.startswith("Voulez-vous dire"):
        payload.update(yes_no_keyboard)

    r = requests.post(url, json=payload)

    return r


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        message = request.get_json()
        chat_id = message['message']['chat']['id']
        txt = message['message']['text']

        send_message(chat_id, make_reply(txt))

        return Response('Ok',
                        status=200)  # this will prevent telegram from spamming us with requets (200 status means we received the goddamn message from telegram)
    else:
        return '<h1>TMPA CHATBOT YEYY</h1>'


# update_id = None
# while True:
#    print("...")
#    updates = bot.get_updates(offset=update_id)
#    updates = updates['result']
#    if updates:
#        for item in updates:
#            update_id = item["update_id"]
#            try:
#                message = item["message"]["text"]
#            except:
#                message = None
#            from_ = item["message"]["from"]["id"]
#            reply = make_reply(message)
#            bot.send_message(reply, from_)


if __name__ == '__main__':
    app.run(debug=True)

# https://api.telegram.org/bot1817211536:AAEfapEpxDzUpZI5qVVT49_F300RQ_5_0hI/setWebhook?url=https://e5d40887917d.ngrok.io