# THIS IS THE USAGE API:
# IT WILL CALL THE MODEL WE TRAINED TO MAKE PREDICTION

# IMPORT PACKAGES
import PreProcessing as pp
import os
from keras.models import load_model
import utils
import pandas as pd
import sys

#==========================================
# Name: make_prediction
# Purpose: CALL TRAINED MODE AND MAKE PREDICTION ON USER'S NEW DATA
# Input Parameter(s): target_file_path --- THE PATH OF USER'S NEW DATA
# Return Value(s): NONE
#============================================
def make_prediction(target_file_path):
    # SET UP FILE PATH
    name_list = target_file_path.split('/')
    pure_name = name_list[-1]
    pure_path = target_file_path.replace(pure_name,'')
    target_clean_file = pure_path + 'clean_' + pure_name
    predict_file = pure_path + 'predict_result_' + pure_name

    # CLEAN DATA
    pp.clean_data(target_file_path, target_clean_file, mode='test')
    target_sequences = pp.convert_text2sequences(extrafilepath=target_clean_file,mode='test')

    # MAKE PREDICTION
    try:
        model_file = os.listdir('model/')
        model = load_model('model/' + model_file[-1])
        y_pred = model.predict([target_sequences[1], target_sequences[2], target_sequences[3]])
    except:
        print('Model cannot be found!')

    # SAVE PREDICTION RESULT
    result_dict = {'turn1':[], 'turn2':[], 'turn3':[], 'label':[]}
    target_df = pd.read_csv(target_file_path)
    for i in range(len(y_pred)):
        cur_record = target_df.iloc[i,:]
        cur_pred = y_pred[i].tolist()
        cur_lab = cur_pred.index(max(cur_pred))
        result_dict['turn1'].append(cur_record['turn1'])
        result_dict['turn2'].append(cur_record['turn2'])
        result_dict['turn3'].append(cur_record['turn3'])
        result_dict['label'].append(utils.label2emotion[cur_lab])
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df.to_csv(predict_file, index=False)
    print('Prediction file has been generated successfully!')


if __name__ == '__main__':
    """
    target_file_path = 'testdata/test.csv'
    make_prediction(target_file_path)
    """
    if len(sys.argv) != 2:
        print('The numbers of arguments are not correct!\n')
        print('Following this instruction:\n')
        print('python Useit.py Your_File_Dir\n')
    else:
        target_file_path = sys.argv[1]
        make_prediction(target_file_path)

    pass