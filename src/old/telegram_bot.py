import telebot
import os
import linguistic_analysis
from text_parse import parse_file


token = os.getenv('bot_token')

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Hello! I am your friendly bot. How can I assist you today?")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    rows = parse_file("example.csv", 1, 1000)
    result = linguistic_analysis.analyse_tfidf_cosine(rows, [message.text, 'hello world'], True)
    bot.reply_to(message, result[0])

bot.polling()