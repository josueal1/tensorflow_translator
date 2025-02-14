import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # This is required for the tokenizer

# Load the pre-trained model from TensorFlow Hub
# Here we use the TensorFlow Hub model for English-to-Spanish translation
model_url = "https://tfhub.dev/google/translation/english_to_spanish/1"
translator = hub.load(model_url)

def translate_text(input_text):
    # TensorFlow requires input in the form of a tensor
    input_tensor = tf.convert_to_tensor([input_text])

    # Perform translation using the model
    translated_tensor = translator(input_tensor)

    # The output is a tensor of shape (1, 1), so extract the text
    translated_text = translated_tensor.numpy()[0].decode("utf-8")

    return translated_text

# Example usage
english_text = "Hello, how are you?"
translated_text = translate_text(english_text)
print(f"Original (English): {english_text}")
print(f"Translated (Spanish): {translated_text}")
