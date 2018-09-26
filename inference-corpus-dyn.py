import tensorflow as tf
import numpy as np
import os, sys

import params as pm

# Command to split the corpus in subcorpora with maximum 10^5 rows
# $: split -d -l 10000 input_file ./split/split_

# Usage
# $: python3 inference-corpus-dyn.py <model-name> <path-to-corpus-infer> <>
# $: python3 inference-corpus-dyn.py calendar-lstm-dyn-glove /home/asr/Data/classif_task/data_to_filter/split/


def load_sentences_to_infer(directory, text_name):
    data = {}
    data["sentence"] = []
    l1 = 0
    with tf.gfile.GFile(os.path.join(directory, text_name), "rb") as f:
            # strip() removes white spaces before and after the string
            # decode() converst a byte object ('b) in a python3 string
            list_of_sentences = [s.strip().decode() for s in f.readlines()]
            num_rows_1 = len(list_of_sentences)
            for i in range(num_rows_1):
                data["sentence"].append(list_of_sentences[i])
    return data


def get_sentence_by_class(infer_out, class_to_filter = 1):
    indx = lambda cl: np.where(infer_out['class'].astype(int) == cl)[0]
    return [sentences_to_infer['sentence'][i] for i in indx(class_to_filter)]

def get_sentence_by_prob(infer_out, sentences, prob_min, prob_max = 1.0):
    indx = lambda p_min, p_max: np.where((infer_out[:,1] > p_min) \
                    & (infer_out[:,1] < p_max))[0]
    return [sentences['sentence'][i] for i in indx(prob_min, prob_max)]


def saving(outF, sentence):
    for line in sentence:
            print(line, file = outF)

def filtering(directory, SAVING_PATH, prob_min, prob_max = 1.0):
    #title_file = SAVING_PATH+MODEL_NAME+'-'+str(datetime.datetime.today())[:19].replace(" ", "_").replace(":", "_")+".txt"
    title_file = SAVING_PATH+MODEL_NAME+'-'+str(prob_min)+".txt"
    outF = open(title_file, "w")
    for filename in os.listdir(directory):
        sentences_to_infer = load_sentences_to_infer(directory, str(filename))
        inference_output = predictor_fn(sentences_to_infer)
        print('Prediction done on', filename)
        infer_out = inference_output['probabilities']
        sentence_filtered_tmp = get_sentence_by_prob(infer_out, sentences_to_infer, prob_min, prob_max)
        saving(outF, sentence_filtered_tmp)
        print('----> Saved')
    print('Everything was saved succesfully!')
    outF.close()

if __name__ == "__main__":

    MODEL_NAME = str(sys.argv[1]) # 'calendar-model-04'
    dir_data_to_filter =  str(sys.argv[2]) #'/home/asr/Data/classif_task/data_to_filter/split/'

    # Parameters: select sentences which are classified as class 1 with probability between min and max
    probability_min = 0.98
    probability_max = 1.00

    # Path where the model is saved!
    model_dir = os.path.join(os.getcwd(),'trained_models/{}'.format(MODEL_NAME))
    export_dir = model_dir + "/export/predict/"
    saved_model_dir = export_dir + "/" + os.listdir(path=export_dir)[-1]

    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir = saved_model_dir,
                                                            signature_def_key="prediction")

    # Path where the filtered sentences will be saved
    SAVING_PATH=pm.INFERENCE_DIR
    filtering(dir_data_to_filter, SAVING_PATH, prob_min = probability_min, prob_max = probability_max)
