import tensorflow as tf
from transformers import TFBertModel, BertConfig, TFDistilBertModel, DistilBertConfig, \
TFRobertaModel, RobertaConfig, TFDebertaModel, DebertaConfig
import keras
from keras import layers, Model



def resnet_block(x, filters, kernel_size=1, stride=1, conv_shortcut=True, name=None):
    bn_axis = -1 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    
    if conv_shortcut:
        shortcut = layers.Conv1D(filters, 1, strides=stride, use_bias=False, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x



class CustomEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super(CustomEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=7, output_dim=embed_dim)   

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        token_embeddings = self.token_emb(x)
        return token_embeddings + positions
    

def bert_aa_model(input_shape=(7, 20, 1), embed_dim=64, num_layers=4, num_heads=4, hidden_dim=64,learning_rate=0.0005):
    # Define input
    inputs = keras.layers.Input(shape=input_shape) # Converts the input one-hot encoding to an integer index
    x = tf.squeeze(inputs, axis=-1)  # Remove the last dimension (number of channels)
    input_ids = tf.argmax(x, axis=-1) 

    attention_mask = tf.cast(tf.not_equal(input_ids, 0), dtype=tf.int32)# Generate attention mask
   
    embedding_layer = CustomEmbedding(vocab_size=20, embed_dim=embed_dim) 
    embedded_input = embedding_layer(input_ids)

    # Load the pre-trained BERT model configuration
    config = BertConfig(
        vocab_size=20,  # Suppose the vocabulary size is 20
        hidden_size=embed_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_dim,
        max_position_embeddings=input_shape[0] ,# Suppose the sequence length is 7
        use_pooler=False  # Disable the pooler layer
    )
    # Load the predefined BERT model
    bert = TFBertModel(config)


    bert.bert.embeddings.word_embeddings = keras.layers.Embedding(input_dim=20, output_dim=embed_dim)
    bert.bert.embeddings.position_embeddings = keras.layers.Embedding(input_dim=input_shape[0] , output_dim=embed_dim)

    # Build the input dictionary
    bert_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    x = bert(bert_inputs)[0]  # Gets the hidden state of the last layer

    # Add residual join
    x = resnet_block(x, filters=embed_dim, name='resnet_1')
    x = resnet_block(x, filters=embed_dim, name='resnet_2')

    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)   # Layer normalization
    x = keras.layers.GlobalAveragePooling1D()(x)   # Global average pooling

    # Residual fully connected layer
    fc_residual = layers.Dense(hidden_dim, activation=None)(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Add()([fc_residual, x])
    x = layers.Activation('relu')(x)


    x = keras.layers.Dropout(0.5)(x)


    outputs = layers.Dense(1, activation='linear')(x) 



    model = keras.Model(inputs=inputs, outputs=outputs)
    # Use Adam optimizer and enable gradient cropping
    optimizer = keras.optimizers.Adam(learning_rate, clipvalue=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model


def roberta_aa_model(input_shape=(7, 20, 1), embed_dim=64, num_layers=4, num_heads=4, hidden_dim=64,learning_rate=0.0005):
    # Define inputs
    inputs = keras.layers.Input(shape=input_shape)

    # Convert one-hot encoded input to integer indices
    x = tf.squeeze(inputs, axis=-1)  # Remove the last dimension (channels)
    input_ids = tf.argmax(x, axis=-1)

    # Generate attention mask
    attention_mask = tf.cast(tf.not_equal(input_ids, 0), dtype=tf.int32)

    # Custom embedding layer
    embedding_layer = CustomEmbedding(vocab_size=20, embed_dim=embed_dim)
    embedded_input = embedding_layer(input_ids)

    # Load pre-trained RoBERTa model configuration
    config = RobertaConfig(
        vocab_size=20,  # Assume a vocabulary size of 20
        hidden_size=embed_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_dim,
        max_position_embeddings=input_shape[0]  # Assume sequence length of 7
    )

    # Load pre-trained RoBERTa model
    roberta = TFRobertaModel(config)

    # Replace RoBERTa embedding layers
    roberta.roberta.embeddings.word_embeddings = keras.layers.Embedding(input_dim=20, output_dim=embed_dim)
    roberta.roberta.embeddings.position_embeddings = keras.layers.Embedding(input_dim=input_shape[0], output_dim=embed_dim)

    # Construct input dictionary
    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # Pass inputs to RoBERTa model
    x = roberta(model_inputs)[0]  # Retrieve the last hidden state

    # Global average pooling
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Layer normalization
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # MLP head
    x = keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model




def distilbert_aa_model(input_shape=(7, 20, 1), embed_dim=64, num_layers=4, num_heads=4, hidden_dim=64):
    # Define inputs
    inputs = keras.layers.Input(shape=input_shape)

    # Convert one-hot encoded input to integer indices
    x = tf.squeeze(inputs, axis=-1)  # Remove the last dimension (channels)
    input_ids = tf.argmax(x, axis=-1)

    # Generate attention mask
    attention_mask = tf.cast(tf.not_equal(input_ids, 0), dtype=tf.int32)

    # Custom embedding layer
    embedding_layer = CustomEmbedding(vocab_size=20, embed_dim=embed_dim)
    embedded_input = embedding_layer(input_ids)

    # Load pre-trained DistilBERT model configuration
    config = DistilBertConfig(
        vocab_size=20,  # Assume a vocabulary size of 20
        dim=embed_dim,
        n_layers=num_layers,
        n_heads=num_heads,
        hidden_dim=hidden_dim,
        max_position_embeddings=input_shape[0]  # Assume sequence length of 7
    )

    # Load pre-trained DistilBERT model
    distilbert = TFDistilBertModel(config)

    # Replace DistilBERT embedding layers
    distilbert.distilbert.embeddings.word_embeddings = keras.layers.Embedding(input_dim=20, output_dim=embed_dim)
    distilbert.distilbert.embeddings.position_embeddings = keras.layers.Embedding(input_dim=7, output_dim=embed_dim)

    # Construct input dictionary
    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # Pass inputs to DistilBERT model
    x = distilbert(model_inputs)[0]  # Retrieve the last hidden state

    # Global average pooling
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Layer normalization
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # MLP head
    x = keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model



def deberta_aa_model(input_shape=(7, 20, 1), embed_dim=64, num_layers=4, num_heads=4, hidden_dim=64):
    # Define inputs
    inputs = keras.layers.Input(shape=input_shape)

    # Convert one-hot encoded input to integer indices
    x = tf.squeeze(inputs, axis=-1)  # Remove the last dimension (channels)
    input_ids = tf.argmax(x, axis=-1)

    # Generate attention mask
    attention_mask = tf.cast(tf.not_equal(input_ids, 0), dtype=tf.int32)

    # Custom embedding layer
    embedding_layer = CustomEmbedding(vocab_size=20, embed_dim=embed_dim)
    embedded_input = embedding_layer(input_ids)

    # Load pre-trained DeBERTa model configuration
    config = DebertaConfig(
        vocab_size=20,  # Assume a vocabulary size of 20
        hidden_size=embed_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_dim,
        max_position_embeddings=7  # Assume sequence length of 7
    )

    # Load pre-trained DeBERTa model
    deberta = TFDebertaModel(config)

    # Replace DeBERTa embedding layers
    deberta.deberta.embeddings.word_embeddings = keras.layers.Embedding(input_dim=20, output_dim=embed_dim)
    deberta.deberta.embeddings.position_embeddings = keras.layers.Embedding(input_dim=7, output_dim=embed_dim)

    # Construct input dictionary
    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # Pass inputs to DeBERTa model
    x = deberta(model_inputs)[0]  # Retrieve the last hidden state

    # Global average pooling
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Layer normalization
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # MLP head
    x = keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model
