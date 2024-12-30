import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Input, Dense, LayerNormalization, Dropout, Reshape, Permute, Embedding, Lambda, MultiHeadAttention, GlobalAveragePooling1D
from transformers import TFTransfoXLModel, TransfoXLConfig
import keras

def mlp(x, hidden_units, dropout_rate):
    #Creating a Multilayer Perceptron (MLP)
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def vit_aa_model(input_shape=(7, 20, 1), patch_size=(1, 20), num_heads=4, projection_dim=64, transformer_layers=4, mlp_head_units=[64], learning_rate=0.0005):
    inputs = Input(shape=input_shape)

    # Split the input image into patches
    patches = Patches(patch_size)(inputs)
    
    # Encode the patches
    num_patches = (input_shape[0] // patch_size[0]) * (input_shape[1] // patch_size[1])
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim)(patches)

    # Transformer encoder
    for _ in range(transformer_layers):
        # Layer normalization
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head self-attention
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = Dropout(0.1)(attention_output)
        
        # Skip connection
        x2 = x1 + x2
        
        # Layer normalization
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=mlp_head_units, dropout_rate=0.1)
        encoded_patches = x2 + x3

    # Global average pooling
    representation = GlobalAveragePooling1D()(encoded_patches)
    
    # Layer normalization
    representation = LayerNormalization(epsilon=1e-6)(representation)
    
    # MLP head
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    # Output layer
    outputs = Dense(units=1)(features)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model


#------------------------------------------------
#%% Transformer model 
#------------------------------------------------

def mlp(x, hidden_units, dropout_rate):
    """ 创建一个多层感知机（MLP） """
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Layer normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)  # Multi-head self-attention
    x = Dropout(dropout)(x)
    res = x + inputs  # Skip connection
    x = LayerNormalization(epsilon=1e-6)(res)  # Layer normalization
    x = mlp(x, hidden_units=[ff_dim], dropout_rate=dropout)  # MLP
    return x + res  # Skip connection

def Transformer_aa_model(input_shape=(7, 20, 1), embed_dim=64, num_heads=4, ff_dim=64, transformer_layers=4, mlp_head_units=[64], learning_rate=0.0005):
    inputs = Input(shape=input_shape)

    # Convert the input one-hot encoding to integer indices
    # Assuming the input is (batch_size, 7, 20, 1), we need to convert it to (batch_size, 7, 20)
    x = tf.squeeze(inputs, axis=-1)  # Remove the last dimension (channels)
    x = tf.argmax(x, axis=-1)  # Convert one-hot encoding to integer indices

    # Token embedding and position embedding
    embedding_layer = TokenAndPositionEmbedding(maxlen=input_shape[0], vocab_size=input_shape[1], embed_dim=embed_dim)
    x = embedding_layer(x)

    # Transformer encoder
    for _ in range(transformer_layers):
        x = transformer_encoder(x, head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=0.1)

    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    
    # Layer normalization
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # MLP head
    x = mlp(x, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    # Output layer
    outputs = Dense(units=1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model




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


def transformer_xl_model(input_shape=(7, 20, 1), embed_dim=64, num_layers=4, num_heads=4, hidden_dim=64):
    '''
    Transformer XL is a modified Transformer model that is particularly well suited to handle long sequences.
    It addresses the limitations of traditional Transformers in dealing with long sequences by introducing 
    relative position coding and segment-level looping mechanisms.
    '''
    # Define input
    inputs = keras.layers.Input(shape=input_shape)

    # Convert the input one-hot encoding to integer indices
    x = tf.squeeze(inputs, axis=-1)  # Remove the last dimension (channels)
    input_ids = tf.argmax(x, axis=-1)

    # Generate attention mask
    attention_mask = tf.cast(tf.not_equal(input_ids, 0), dtype=tf.int32)

    # Custom embedding layer
    embedding_layer = CustomEmbedding(vocab_size=20, embed_dim=embed_dim)
    embedded_input = embedding_layer(input_ids)

    # Load pre-trained Transformer XL model configuration
    config = TransfoXLConfig(
        vocab_size=20,  # Assume vocabulary size is 20
        d_model=embed_dim,
        n_layer=num_layers,
        n_head=num_heads,
        d_inner=hidden_dim,
        mem_len=7  # Assume sequence length is 7
    )

    # Load pre-trained Transformer XL model
    transformer_xl = TFTransfoXLModel(config)

    # Replace Transformer XL's embedding layer
    transformer_xl.transformer.word_emb = keras.layers.Embedding(input_dim=20, output_dim=embed_dim)
    transformer_xl.transformer.mem_len = 7

    # Build input dictionary
    model_inputs = {
        'input_ids': input_ids,
        'mems': None  # Transformer XL uses a memory mechanism; here we pass None or previous memory
    }

    # Pass the input dictionary to the Transformer XL model
    x = transformer_xl(model_inputs)[0]  # Get the last layer's hidden states

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
