from fastapi import FastAPI
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle as pkl
import numpy as np

app = FastAPI()

# Load the tokenizer used during training
with open("./model_tokenizer_jokes.pickle", "rb") as file:
    model, tokenizer = pkl.load(file)


@app.get("/generate_joke/")
def generate_joke(seed_text: str):
    """Endpoint to generate jokes

    Args:
        seed_text (str): _description_

    Returns:
        _type_: _description_
    """
    # Preprocess the input seed text
    max_sequence_length = 13
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_length - 1, padding="pre"
    )

    # Use the model to predict the next word
    predicted_word_index = np.argmax(model.predict(token_list), axis=-1)
    predicted_word = tokenizer.index_word[predicted_word_index[0]]

    # Construct the generated joke
    generated_joke = seed_text + " " + predicted_word

    return {"generated_joke": generated_joke}
