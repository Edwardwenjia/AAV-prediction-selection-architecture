import os
import numpy as np
import matplotlib as mpl
from utils.utils_f4f import CustomEarlyStopping
from absl import flags, app
from utils.plot import plot_results, fit_gaussian_mixture
from utils.processing import load_and_preprocess_data, encode_features, define_model, train_model,save_model,evaluate_predictions


def configure_plotting():
    """配置绘图参数"""
    mpl.rcParams['svg.fonttype']  =  'none'
    mpl.rcParams['font.family']  =  ['sans-serif']
    mpl.rcParams['font.sans-serif']  =  ['Arial']
    mpl.rcParams['text.usetex']  =  False
    mpl.rcParams['pdf.fonttype']  =  42
    mpl.rcParams['ps.fonttype']  =  42

# 定义超参数
FLAGS  =  flags.FLAGS

# 数据路径和保存路径
flags.DEFINE_string('file_path', '/public/chenjiawen/AAV/AAV_Prediction_Selection_Architecture/data/1st_780w_packseq_aa.tsv', 'Data file path')
flags.DEFINE_string('save_dir', '/public/chenjiawen/AAV/AAV_Prediction_Selection_Architecture/MP/results/', 'Directory to save results')
flags.DEFINE_string('model_name', 'GPT', 'Name of the model to use (e.g., Bert,Transformer, GPT, Llama2, etc.)')
flags.DEFINE_string('htitle', 'seq', 'Name of the table title (e.g., aa:ADKSNJS, seq:YNKKDKLSMLDNKFFIK.)')
flags.DEFINE_integer('sample_step', 50, 'Number of iterations in training')

# 训练参数
flags.DEFINE_integer('iterations', 200, 'Number of iterations in training')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training')
flags.DEFINE_integer('epochs', 40, 'Number of epochs for training')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for training')


# 模型参数
flags.DEFINE_string('input_shape', '(21,20,1)', 'Input shape of the model (e.g., "aa: (7, 20, 1), seq:(21,20,1)")')  # 将 input_shape 定义为字符串
flags.DEFINE_integer('embed_dim', 64, 'Dimension of token embeddings')
flags.DEFINE_integer('num_layers', 4, 'Number of transformer layers')
flags.DEFINE_integer('num_heads', 4, 'Number of attention heads')
flags.DEFINE_integer('hidden_dim', 64, 'Dimension of the feed-forward network hidden layer')


# 文件名
flags.DEFINE_string('result_file_name', 'results', 'Name of the result file')
flags.DEFINE_string('plot_file_name', '1st_nor_package_prediction_correlation_test', 'Name of the correlation plot file')
flags.DEFINE_string('gm_plot_file_name', '1st_nor_package_prediction_distribution_test', 'Name of the distribution plot file')





# 主函数，用于调用以上定义的各个模块
def main_pipeline(file_path, save_dir,htitle,model_name, input_shape, sample_step , embed_dim, num_layers, num_heads, hidden_dim, batch_size, epochs,learning_rate ,result_file_name, plot_file_name, gm_plot_file_name):
    configure_plotting()
    
    # Load and preprocess data
    df_modeling  =  load_and_preprocess_data(file_path,sample_step,htitle)
    
    # Split data into train, validation, and test sets
    train, validate, test  =  np.split(df_modeling.sample(frac = 1), [int(.8*len(df_modeling)), int(.9*len(df_modeling))])
    
    # Encode features
    train_x  =  encode_features(train)
    train_y  =  train['nor_package'].values
    validate_x  =  encode_features(validate)
    validate_y  =  validate['nor_package'].values
    test_x  =  encode_features(test)
    test_y  =  test['nor_package'].values
    
    # Define and compile the model
    model  =  define_model(model_name,input_shape, embed_dim, num_layers, num_heads, hidden_dim,learning_rate)
    # Train the model
    model, history  =  train_model(model, train_x, train_y, validate_x, validate_y, batch_size, epochs, 
                                 [CustomEarlyStopping(ratio = 0.90, patience = 20, restore_best_weights = True)])
    
    # Save the trained model
    filename_dir  =  os.path.join(save_dir, model_name)
    if not os.path.exists(filename_dir):
        os.makedirs(filename_dir)

    save_model(model, filename_dir, f"model_1st_{model_name}_aa_nor_package")
    
    # Evaluate predictions and plot results
    preds_df, correlation  =  evaluate_predictions(model, test_x, test_y, filename_dir, result_file_name)
    plot_results(preds_df, filename_dir, plot_file_name)
    
    # Fit Gaussian Mixture and plot
    fit_gaussian_mixture(preds_df, 2, filename_dir, gm_plot_file_name)



def parse_input_shape(input_shape_str):
    """将字符串形式的输入形状解析为元组"""
    try:
        # 使用 eval 解析字符串为元组
        parsed_shape  =  eval(input_shape_str)
        if not isinstance(parsed_shape, tuple):
            raise ValueError("Input shape must be a tuple.")
        return parsed_shape
    except Exception as e:
        raise ValueError(f"Invalid input shape format: {input_shape_str}. Error: {e}")




def main(_):
    # 从 FLAGS 中读取超参数

    input_shape_str  =  FLAGS.input_shape
    input_shape  =  parse_input_shape(input_shape_str)  # 解析 input_shape
    plot_file_name = f'{FLAGS.plot_file_name}_{FLAGS.htitle}'
    gm_plot_file_name = f'{FLAGS.gm_plot_file_name}_{FLAGS.htitle}'
    # 调用 main_pipeline 函数
    main_pipeline(
        file_path = FLAGS.file_path,
        save_dir = FLAGS.save_dir,
        htitle = FLAGS.htitle,
        model_name = FLAGS.model_name,
        input_shape = input_shape,
        sample_step =  FLAGS.sample_step,
        embed_dim = FLAGS.embed_dim,
        num_layers = FLAGS.num_layers,
        num_heads = FLAGS.num_heads,
        hidden_dim = FLAGS.hidden_dim,
        batch_size = FLAGS.batch_size,
        learning_rate = FLAGS.learning_rate,
        epochs = FLAGS.epochs,
        result_file_name = FLAGS.result_file_name,
        plot_file_name = plot_file_name,
        gm_plot_file_name = gm_plot_file_name
    )

if __name__  ==  '__main__':
    app.run(main)