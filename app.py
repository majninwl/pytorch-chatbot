#!/usr/bin/env python
# pylint: disable=C0116
# This program is dedicated to the public domain under the CC0 license.

"""
First, a few callback functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Example of a bot-user conversation using ConversationHandler.
Send /start to initiate the conversation.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import io
import json
import train
import random
import torch

from spellchecker import SpellChecker
from typing import Dict

from model import RNN
from nltk_utils import bag_of_words, tokenize
import configparser as cfg

from telegram import ReplyKeyboardMarkup, Update, ReplyKeyboardRemove
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    PicklePersistence,
    CallbackContext,
)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

ASKING, TYPING_FIRST_CHOICE, TYPING_SECOND_CHOICE = range(3)

reply_keyboard = [
    ['Oui', 'Non']
]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)


def start(update: Update, context: CallbackContext) -> int:
    """Start the conversation, display any stored data and ask user for input."""
    reply_text = (
        "Bonjour! Mon nom est TMPA Bot,"
        "J'aimerai être util pour vous, comment je peux vous aidez!"
    )
    update.message.reply_text(reply_text)

    return ASKING


def handle_question(update: Update, context: CallbackContext) -> int:
    """Ask the user for info about the selected predefined choice."""
    text = update.message.text.lower()
    context.user_data['question'] = text
    context.user_data['opened_conversation'] = True
    possible_answers = respond(text)
    if (len(possible_answers) == 0):
        reply_text = (
            f'Veuillez reformuler votre question, car je ne arrive pas à trouvez une question valide pour question :('
        )
        update.message.reply_text(reply_text)
        return ASKING
    else:
        context.user_data['answers'] = possible_answers
        first_question = possible_answers[0]['patterns'][0];
        reply_text = f'Voulez vous dire: {first_question}'
        update.message.reply_text(reply_text, reply_markup=markup)
        return TYPING_FIRST_CHOICE


def handle_first_choice(update: Update, context: CallbackContext) -> int:
    """Ask the user for a description of a custom category."""
    text = update.message.text.lower()
    possible_answers = context.user_data['answers']
    if (text == 'oui'):
        first_intent = possible_answers[0]
        reply_text = first_intent['responses'][0]
        update.message.reply_text(reply_text)
        return ConversationHandler.END
    else:
        second_question = possible_answers[1]['patterns'][0];
        reply_text = f'Voulez vous dire: {second_question}'
        update.message.reply_text(reply_text, reply_markup=markup)
        return TYPING_SECOND_CHOICE


def handle_second_choice(update: Update, context: CallbackContext) -> int:
    """Ask the user for a description of a custom category."""
    text = update.message.text.lower()
    possible_answers = context.user_data['answers']
    if (text == 'oui'):
        second_intent = possible_answers[1]
        reply_text = second_intent['responses'][0]
        update.message.reply_text(reply_text)
        return ConversationHandler.END
    else:
        reply_text = 'Je n\'ai pas compris votre question alors!!, Veuillez reformuler votre question SVP ;)'
        update.message.reply_text(reply_text)
        return ASKING


def done(update: Update, context: CallbackContext) -> int:
    """Display the gathered info and end the conversation."""
    if (context.user_data['opened_conversation'] == True):
        reply_text = 'Bye!'
        update.message.reply_text(reply_text)
    return ConversationHandler.END


def main() -> None:
    """Run the bot."""
    # Create the Updater and pass it your bot's token.
    persistence = PicklePersistence(filename='conversationbot')
    token = read_token_from_config_file("config.cfg")
    updater = Updater(token, persistence=persistence)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states ASKING, TYPING_CHOICE and TYPING_REPLY
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ASKING: [
                MessageHandler(
                    Filters.text & ~(Filters.regex(r'(?i)^(bye|good bye|au revoir)$') | Filters.command('stop')),
                    handle_question
                ),
                MessageHandler(Filters.regex(r'(?i)^(bye|good bye|au revoir)$') | Filters.command('stop'), done),
            ],
            TYPING_FIRST_CHOICE: [
                MessageHandler(
                    Filters.text & ~(Filters.regex(r'(?i)^(bye|good bye|au revoir)$') | Filters.command('stop')),
                    handle_first_choice
                )
            ],
            TYPING_SECOND_CHOICE: [
                MessageHandler(
                    Filters.text & ~(Filters.regex(r'(?i)^(bye|good bye|au revoir)$') | Filters.command('stop')),
                    handle_second_choice
                )
            ],
        },
        fallbacks=[MessageHandler(Filters.regex(r'(?i)^(bye|good bye|au revoir)$') | Filters.command('stop'), done)],
        name="my_conversation",
        persistent=True,
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


def init_chatbot():
    global data
    global tags
    global model
    global all_words
    global device
    global indexed_intents
    indexed_intents = {}
    FILE = "data.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with io.open('intents.json', 'r', encoding='utf8') as json_data:
        intents = json.load(json_data)
        for intent in intents['intents']:
            indexed_intents[intent["tag"]] = intent

    data = torch.load(FILE)
    tags = data['tags']

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    model_state = data["model_state"]

    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(model_state)
    model.eval()


def respond(user_input):
    sentence = user_input
    sentence = tokenize(sentence)

    # Keywords
    non_spellchecked = ['amp', 'ampe', 'ampi', 'remote', 'morocco', 'foodex', 'colisage']

    spell = SpellChecker(language='fr')

    sentence_corrected = []

    for word in sentence:
        if word in non_spellchecked or word.lower() in non_spellchecked:
            sentence_corrected.append(word)
        else:
            sentence_corrected.append(spell.correction(word))

    logger.info("original sentence:%s", sentence)
    logger.info("corrected sentence:%s", sentence_corrected)
    sentence = sentence_corrected

    # print(sentence)

    sentence_bag_of_words = bag_of_words(sentence, all_words)
    sentence_bag_of_words = sentence_bag_of_words.reshape(1, sentence_bag_of_words.shape[0])
    sentence_bag_of_words = torch.from_numpy(sentence_bag_of_words).to(device)

    output = model(sentence_bag_of_words)
    # Prediction with highest score
    _, predicted = torch.max(output, dim=1)
    # All predictions sorted from highest score to lowest
    _, predictions = torch.sort(output, dim=1, descending=True)

    # Get the tag of the assert with highest prediction score
    tag = tags[predicted.item()]
    tag2 = tags[predictions[0].numpy()[1]]

    # i=1
    ##?????
    # directly_answered = ['Salutations', 'au revoir', 'merci', 'contact']
    ##Searching for the first tag that is not salutations and make it our second tag weird??!!
    # while True:
    #    tag2 = tags[predictions[0].numpy()[i]]
    #    i += 1
    #    if(tag2 not in directly_answered):
    #        break

    # get the probability of all tags sorted by tag Index
    probs = torch.softmax(output, dim=1)
    # get the probability of the predicted answer which should be the highest probability
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        first_high_prob_intent = indexed_intents[tag]
        second_high_prob_intent = indexed_intents[tag2]
        return [first_high_prob_intent, second_high_prob_intent]
    else:
        return []


def read_token_from_config_file(config):
    parser = cfg.ConfigParser()
    parser.read(config)
    return parser.get('creds', 'token')


if __name__ == '__main__':
    init_chatbot()
    main()