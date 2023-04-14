import pickle

task_names = ['sst2', 'stsb', 'qnli']
frame_names=['input_x_gradients', 'deeplift', 'kernel_shap', 'lime', 'guided_backprop', 'integrated_gradients']

process_suffix = '_processed'

plausibility_name  = 'plausibility'
simplicity_name = 'simplicity'
reproducibility_name  = 'reproducibility'
term_names = [plausibility_name, simplicity_name, reproducibility_name]
              
IQS_alphas_graph_file = 'IQS_alphas_graph_file'
    
def path_to_task_files(task_name):
    return f'../{task_name}/Explain/analysis_files'

def load_processed_out_and_res_files_for_task(task_name):
    processed = {}
    task_files_path = path_to_task_files(task_name)
    for frame_name in frame_names:
        processed[frame_name] = {}
        with open(f'{task_files_path}/{frame_name}_out_processed.pkl', 'rb') as f:
            processed[frame_name]['out'] = pickle.load(f)

        with open(f'{task_files_path}/{frame_name}_results_processed.pkl', 'rb') as f:
            processed[frame_name]['results'] = pickle.load(f)

    return processed

def load_processed_out_and_res_files_for_all_task():
    all_task_dict = {}
    for task_name in task_names:
        processed_out = load_processed_out_and_res_files_for_task(task_name)
        all_task_dict[task_name] = {}
        all_task_dict[task_name]= processed_out

    return all_task_dict

def load_calculated_term(term_file):
    ret_dict = {}
    with open(f'{term_file}.pkl', 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict

def load_all_calculated_terms_for_tasks():
    all_term_dict = {}
    for term_name in term_names:
        calcualted_term = load_calculated_term(term_name)
        all_term_dict[term_name] = calcualted_term
    return all_term_dict