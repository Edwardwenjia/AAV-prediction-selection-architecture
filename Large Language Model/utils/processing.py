
from utils.utils_f4f import AA_hotencoding
from model.BERT import bert_aa_model, roberta_aa_model
from model.Transformer import vit_aa_model
from model.GPT import gpt2_model
from model.ResNet import resnet_aa_model
from model.CTRL import ctrl_model
from model.U_Net import mobilenetv2_aa_model
from model.Inception import inception_aa_model
import numpy as np 
import pandas as pd
import os


def load_and_preprocess_data(file_path, sample_step = 500, htitle='aa'):
    # Load and preprocess data
    df_all  =  pd.read_csv(file_path, delimiter = '\t')
    df_sampled  =  df_all.iloc[::sample_step]
    if htitle=='aa':
        df_modeling  =  df_sampled.rename(columns = {'aa': 'AA_sequence'})
        df_modeling['label']  =  0
        df_modeling  =  df_modeling[['AA_sequence', 'nor_package', 'label']]
        df_modeling  =  df_modeling[~df_modeling['nor_package'].isnull() & ~np.isinf(df_modeling['nor_package'])]
        df_modeling.reset_index(drop = True, inplace = True)
    else:
        df_modeling  =  df_sampled.rename(columns = {'seq': 'AA_sequence'})
        df_modeling['label']  =  0
        df_modeling  =  df_modeling[['AA_sequence', 'nor_package', 'label']]
        df_modeling  =  df_modeling[~df_modeling['nor_package'].isnull() & ~np.isinf(df_modeling['nor_package'])]
        df_modeling.reset_index(drop = True, inplace = True)
    return df_modeling



def encode_features(df, aa_col = 'AA_sequence'):
    # one-hot Encoding of amino acid sequence
    encoded_x  =  np.asarray([AA_hotencoding(variant) for variant in df[aa_col]])
    return encoded_x


def define_model(model_name, input_shape, embed_dim, num_layers, num_heads, hidden_dim,learning_rate):
    """定义BERT模型"""
    if model_name == 'ResNet':
        model = resnet_aa_model(input_shape=input_shape, L1=input_shape[0]*input_shape[1], L2=input_shape[1], learning_rate= learning_rate)
    elif model_name =='UNet':      
        model = mobilenetv2_aa_model(input_shape=input_shape, learning_rate= learning_rate)

    elif model_name =='Inception':      
        model = inception_aa_model(input_shape=input_shape, L1=input_shape[0]*input_shape[1], L2=input_shape[1], learning_rate= learning_rate)

    elif model_name  == 'Bert':
        model  =  bert_aa_model(input_shape = input_shape, embed_dim = embed_dim, num_layers = num_layers, num_heads = num_heads, hidden_dim = hidden_dim,learning_rate=learning_rate)

    elif model_name =='Transformer':
        model = vit_aa_model(input_shape=input_shape, patch_size=(1, input_shape[1]), num_heads=num_heads, projection_dim=embed_dim, transformer_layers=num_layers, mlp_head_units=[hidden_dim],learning_rate = 0.0005)

    elif model_name == 'CTRL':
        model = ctrl_model(input_shape = input_shape, embed_dim = embed_dim, num_layers = num_layers, num_heads = num_heads, hidden_dim = hidden_dim,learning_rate=learning_rate)

    elif model_name == 'GPT':
        model = gpt2_model(input_shape=input_shape, embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim)
    
    else:
        raise ValueError('Model not Defined!')
    
    return model

def train_model(model, train_x, train_y, validate_x, validate_y, batch_size, epochs, callbacks):
    #  Training model  
    history  =  model.fit(train_x, train_y, batch_size = batch_size, epochs = epochs,
                        validation_data = (validate_x, validate_y), verbose = 2, callbacks = callbacks)
    return model, history




def save_model(model, save_dir, model_name):
    # Save the trained model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(os.path.join(save_dir, f'{model_name}.h5'))

def evaluate_predictions(model, test_x, test_y, save_dir, result_file_name):
    # Evaluation model prediction results
    y_pred  =  model.predict(test_x).flatten()
    preds_df  =  pd.DataFrame({'y_pred': y_pred, 'y_true': test_y})
    preds_df.to_csv(os.path.join(save_dir, f'{result_file_name}.csv'), index = False)
    correlation  =  preds_df['y_pred'].corr(preds_df['y_true'])
    return preds_df, correlation
