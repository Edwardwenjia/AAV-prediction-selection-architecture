from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig,LlamaModel
import tensorflow as tf
from tensorflow import keras

class CustomEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super(CustomEmbedding, self).__init__()
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=7, output_dim=embed_dim)  # 假设序列长度为7

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

from transformers import AutoTokenizer, TFLlamaModel

def llama1b_aa_model(input_shape=(7, 20, 1), embed_dim=64, num_layers=4, num_heads=4, hidden_dim=64, learning_rate=0.0005):
    # Define input
    inputs = keras.layers.Input(shape=input_shape)

    # Convert one-hot encoding to integer indices
    x = tf.squeeze(inputs, axis=-1)  # Remove the last dimension (channel)
    input_ids = tf.argmax(x, axis=-1)

    # Generate attention mask
    attention_mask = tf.cast(tf.not_equal(input_ids, 0), dtype=tf.int32)

    # Load pretrained LLaMA tokenizer and model
    model_name = "models/huggyllama/llama1b/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llama_model = TFLlamaModel.from_pretrained(model_name)

    # Create input dictionary for the LLaMA model
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    # Pass input through LLaMA model
    x = llama_model(**model_inputs).last_hidden_state  # Last hidden state

    # Global average pooling
    x = keras.layers.GlobalAveragePooling1D()(x)

    # Layer normalization
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Fully connected MLP head
    x = keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    # Output layer
    outputs = keras.layers.Dense(1)(x)

    # Compile the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    return model
