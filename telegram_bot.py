import os
import re
import telebot
from telebot import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
API_KEY = os.getenv('GOOGLE_API_KEY')

# Initialize bot and Google AI
bot = telebot.TeleBot(TOKEN)
genai.configure(api_key=API_KEY)

# Define the book data and prompt templates
# Define the book data and prompt templates
book_data = {
    "Mankutimmana Kagga": {
        "prompt_template": """You are a friendly Telegram user who loves discussing the Mankutimmana Kagga, an ancient Kannada text. Respond to questions in a polite, engaging way using Telegram-style formatting (bold, emojis, bullet points). Follow these guidelines:

1. Start with a friendly greeting using an appropriate emoji and address the user by their username.
2. Analyze the context from the Mankutimmana Kagga text.
3. Identify key information for the question.
4. Provide a conversational answer using the context. If info is missing, just say so casually.
5. Use *bold* for important concepts, bullet points for lists, and emojis to make it lively.
6. Keep it concise and on-point.
7. If possible, add interesting tidbits from the Mankutimmana Kagga.
8. Ensure your answer fits the question and context.

Format your response using Telegram styles like this:
- Use *asterisks* for bold text
- Start bullet points with a bullet symbol and a space
- Add relevant emojis

Context: {context}
Question: {question}
Friendly Telegram Response: @username""",
    },

    "Mahabharata": {
        "prompt_template": """You're a cool Telegram user who's super into the Mahabharata, the epic Indian tale. When answering questions, keep it casual and use Telegram-style formatting (bold, emojis, bullet points). Here's how to rock it:

1. Kick off with a friendly hey and a fitting emoji.
2. Dive into the Mahabharata context you've got.
3. Pinpoint the juicy bits that'll answer the question.
4. Break down your answer into bite-sized chunks. If you're missing some info, no biggie—just say so.
5. Sprinkle in *bold text* for key ideas, pop in some bullet points, and don't skimp on those emojis!
6. Keep things flowing and make sure it all ties back to the question.
7. If you can, throw in a mini-story or some extra Mahabharata goodness to spice it up.
8. Double-check that you're actually answering the question with what you've got.

Format your response like this:
- Use *asterisks* for bold text
- Start bullet points with a bullet symbol and a space
- Add relevant emojis

Context: {context}
Question: {question}
Epic Telegram Response:""",
    },
}

# Global variable to store the user's current book selection
user_book_selection = {}

def escape_markdown(text):
    text = text.replace('\\n', '\n')
    markdown_chars = r'[\*_\[\]()~`>#\+\-=|{}\.!]'
    escaped_text = re.sub(markdown_chars, lambda m: '\\' + m.group(0), text)
    return escaped_text

def load_vector_store(book_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=f"chroma_db/{book_name}", embedding_function=embeddings)
    return vector_store

def get_qa_chain(vector_store, prompt_template):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Initialize vector stores
vector_stores = {}
for book_name in book_data.keys():
    vector_stores[book_name] = load_vector_store(book_name)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    book_buttons = [types.KeyboardButton(book) for book in book_data.keys()]
    markup.add(*book_buttons)
    
    welcome_message = (
        "Welcome to Granthasaara! 📚\n\n"
        "I can help you explore the following books:\n"
        "- Mankutimmana Kagga\n"
        "- Mahabharata\n\n"
        "Available commands:\n"
        "/start - Show this welcome message\n"
        "/help - Show this welcome message\n"
        "/change_book - Change the current book\n\n"
        "To get started, please select a book from the buttons below. "
        "Once you've selected a book, you can ask me any question about it!"
    )
    
    bot.reply_to(message, welcome_message, reply_markup=markup)

@bot.message_handler(func=lambda message: message.text in book_data.keys())
def select_book(message):
    user_book_selection[message.from_user.id] = message.text
    confirmation_message = (
        f"📘 Great! You've selected '{message.text}'.\n\n"
        "You can now ask me any question about this book. For example:\n"
        "- Who are the main characters?\n"
        "- What is the central theme?\n"
        "- Tell me about an important event.\n\n"
        "What would you like to know?"
    )
    bot.reply_to(message, confirmation_message)

@bot.message_handler(commands=['change_book'])
def change_book(message):
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    book_buttons = [types.KeyboardButton(book) for book in book_data.keys()]
    markup.add(*book_buttons)
    
    change_book_message = (
        "Certainly! Let's change the book. 📖\n\n"
        "Please select a new book from the options below:"
    )
    
    bot.reply_to(message, change_book_message, reply_markup=markup)

@bot.message_handler(func=lambda message: True)
def answer_question(message):
    user_id = message.from_user.id
    if user_id not in user_book_selection:
        bot.reply_to(message, "Please select a book first using the /start command.")
        return

    selected_book = user_book_selection[user_id]
    vector_store = vector_stores[selected_book]
    prompt_template = book_data[selected_book]["prompt_template"]
    qa_chain = get_qa_chain(vector_store, prompt_template)

    try:
        response = qa_chain({"query": message.text})
        response_text = escape_markdown(response['result'])
        bot.reply_to(message, response_text, parse_mode='MarkdownV2')
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {e}")

print("Bot is running...")
bot.polling()