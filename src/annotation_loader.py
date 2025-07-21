import pandas as pd 
from glob import glob 
from pathlib import Path
from src.utils import read_json, read_jsonlines
import re, json


batch_convid_path = "../data/batch_ids.json"

# change these if you wanna load conversations from a different folder 

# for the original phase 1 experiments 
# phase1_meta_annotation_path = "/fs/clip-political/rupak/common_ground/experiments/phase1_experiments/phase1_annotation_output/annotations_metadata"

# for choosing the best annotators as promised in the paper
phase1_meta_annotation_path = "../data/annotation_output/annotations_metadata"
phase1_convos_path = "../data/annotation_output"


def can_be_cast_to_int(value):
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def complete_dict(d: dict): 
    # if the value of a key is unsorted, sort it 
    # if there is a number missing in between, complete it 
    # for example, if the value is [3, 5] make it [3, 4, 5]
    for key, value_list in d.items(): 
        #value_list.sort()
        #full_range = list(range(value_list[0], value_list[-1] + 1))    
        d[key] = sorted(value_list)

    return d

def clean_dict(d: dict): 
    # remove keys that start with "Unnamed" 
    return {key: str(value).strip() for key, value in d.items() if not key.startswith("Unnamed")}    


class Phase1Loader: 
    def __init__(self): 
        batch_to_convo_ids = read_json(batch_convid_path)
        self.batch_to_convo_ids = {batch[5:]: convo_ids for batch, convo_ids in batch_to_convo_ids.items()}

        self.convo_id_to_batch = {convo_id: batch for batch, convo_ids in self.batch_to_convo_ids.items() 
                                  for convo_id in convo_ids}

    # util functions to list batches and convos 
    def list_all_batches(self): 
        return list(self.batch_to_convo_ids.keys())

    def list_all_convo_ids(self):
        return list(self.convo_id_to_batch.keys()) 

    def get_convos_of_length(self, length):
        length = str(length)
        return [convo_id for convo_id in self.list_all_convo_ids() if convo_id.split(".")[0] == length]

    def get_convos_in_batch(self, batchname): 
        batchname = str(batchname)
        return self.batch_to_convo_ids[batchname]
    
    def get_batch_of_convo(self, convo_id):
        return self.convo_id_to_batch[convo_id]
    
    # gather annotations from data 
    def get_meta_annotation_df(self, annotation_batch): 
        meta_annotation_info = glob(f"{phase1_meta_annotation_path}/*batch{annotation_batch}.csv")


        if len(meta_annotation_info) == 0 or len(meta_annotation_info) > 1:
            raise ValueError(f"Found {len(meta_annotation_info)} files for batch {annotation_batch}")
        
        meta_annotation_info = meta_annotation_info[0]
        df = pd.read_csv(meta_annotation_info,dtype={"Conversation ID": str} )
        df.columns = df.columns.str.strip()  # Strip any leading/trailing whitespace

        # Create a dictionary to map old column names to new names
        new_column_names = {
            'Conversation ID': 'conversation_id',
            'Time Taken': 'time_taken',
            'Conversational Friction (Y/N)': 'conversational_friction',
            'Conversational Success': 'conversational_success',
            'Ending Type': 'ending_type'
        }

        # Rename the columns
        df = df.rename(columns=new_column_names)

        return df 
    
    def get_meta_metrics(self, conv_id):
        annotation_batch = self.get_batch_of_convo(conv_id)
        annotation_df = self.get_meta_annotation_df(annotation_batch)

        return annotation_df[annotation_df["conversation_id"] == conv_id].to_dict(orient="records")[0]

    def get_annotated_convo(self, convo_id):
        annotation_batch = self.get_batch_of_convo(convo_id)
        annotation_convo_path = glob(f"{phase1_convos_path}/batch{annotation_batch}_*/{convo_id}.xlsx")
        if len(annotation_convo_path) == 0 or len(annotation_convo_path) > 1:
            raise ValueError(f"Found {len(annotation_convo_path)} files for convo {convo_id}")
        
        annotation_convo_path = annotation_convo_path[0]
        df = pd.read_excel(annotation_convo_path)

        return df

    def get_annotation_metadata(self, convo_id): 
        annotation_batch = self.get_batch_of_convo(convo_id)
        annotation_df = self.get_meta_annotation_df(annotation_batch)

        row_dict =  annotation_df[annotation_df["conversation_id"] == convo_id].to_dict(orient="records")
        
        if not row_dict:
            raise ValueError(f"Conversation {convo_id} not found in batch {annotation_batch}")
        
        return clean_dict(row_dict[0])

    def get_annotation_data(self, convo_id): 
        annotation_df = self.get_annotated_convo(convo_id)
        annotation_metadata = self.get_annotation_metadata(convo_id)

        if annotation_metadata['conversational_friction'] == 'No': 
            return {
                "conversation_df": annotation_df,
                "annotation_explanation_dict": {},
                "metadata": annotation_metadata,
                "friction_turns": {}, 
             
            }

        # print all cells of the column "Conversational Friction" where the value is not NaN
        filtered_df = annotation_df.loc[annotation_df['Conversational Friction'].apply(can_be_cast_to_int),
                                        ['Conversational Friction', 'turn_id', 'Explanation']]
        
        filtered_df['Conversational Friction'] = filtered_df['Conversational Friction'].astype(int)
        
        annotation_dict = filtered_df.groupby('Conversational Friction')['turn_id'].apply(list).to_dict()
        annotation_explanation_dict = filtered_df.groupby('turn_id')['Explanation'].apply(list).to_dict()

        annotation_dict = complete_dict(annotation_dict)

        return {
            "conversation_df": annotation_df,
            "annotation_explanation_dict": annotation_explanation_dict,
            "metadata": annotation_metadata,
            "friction_turns": annotation_dict
        }


class Phase1OutputLoader: 
    def __init__(self, filepath): 
        self.phase1_loader = Phase1Loader()

        # check if the filepath exists, if not, raise an error
        if not Path(filepath).exists(): 
            raise ValueError(f"File {filepath} does not exist")

        self.raw_outputs = read_jsonlines(filepath)
        keys = self.raw_outputs[0].keys()
        
        # hacky, change this later
        if "conversation_id" in keys:
            self.conv_id_to_output = {output["conversation_id"]: output for output in self.raw_outputs}

        elif "convo_id" in keys: 
            self.conv_id_to_output = {output["convo_id"]: output for output in self.raw_outputs}
        
    def get_output_of_convo(self, convo_id): 
        return self.conv_id_to_output[convo_id]
    
    def get_output_of_batch(self, batchname):
        convo_ids = self.phase1_loader.get_convos_in_batch(batchname)
        return {convo_id: self.conv_id_to_output[convo_id] for convo_id in convo_ids}
    
def intervals_intersect(interval1, interval2):
    """Check if two non contiguous intervals intersect."""
    return len(set(interval1).intersection(set(interval2))) > 0

def calculate_overlap(human_response_intervals, model_response_intervals):
    """
    Calculate precision, recall, and F1 score based on the overlap between 
    human response intervals and model response intervals.
    
    Parameters:
    human_response_intervals (list of list): List of interval(s) representing human responses.
    model_response_intervals (list of list): List of interval(s) representing model responses.
    
    Returns:
    dict: A dictionary containing 'overlap', 'precision', 'recall', and 'f1' scores.
    """
    
    overlap = 0

    for model_interval in model_response_intervals:
        for human_interval in human_response_intervals:
            if intervals_intersect(model_interval, human_interval):
                overlap += 1
                break

    total_human_intervals = len(human_response_intervals)
    total_model_intervals = len(model_response_intervals)
    
    recall = overlap / total_human_intervals if total_human_intervals > 0 else 0
    precision = overlap / total_model_intervals if total_model_intervals > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "overlap": overlap,
        "total_human_intervals": total_human_intervals,
        "total_model_intervals": total_model_intervals,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def extract_friction_details_corrected(text):
    # Pattern to match the JSON structure specifically for the desired keys
    pattern = re.compile(
        r'"friction_present":\s*(true|false)|'           # Match "friction_present": true/false
        r'"friction(\d+)":\s*\[([^\]]+)\]|'              # Match "frictionN": [x, y]
        r'"explanation(\d+)":\s*"((?:[^"]|\\")*?)"',     # Match "explanationN": including any " escaped with \"
        re.DOTALL
    )

    extracted_data = {}

    for match in pattern.finditer(text):
        if match.group(1) is not None:
            # Extracting the presence of friction
            extracted_data["friction_present"] = match.group(1) == "true"
        elif match.group(2) is not None:
            # Extracting the friction index and values
            index = match.group(2)
            # Handle each item as a string rather than trying to convert to int
            items = [x.strip().strip('"') for x in match.group(3).split(",")]
            extracted_data[f"friction{index}"] = items
        elif match.group(4) is not None:
            # Extracting explanations
            index = match.group(4)
            explanation = match.group(5).replace('\\"', '"')  # Correct escaped quotes
            extracted_data[f"explanation{index}"] = explanation

    # Do some post processing to convert the friction values to integers
    # sometimes, the model will say None, and sometimes it wil say Turn X instead of X 

    to_remove_keys = []
    for key, value in extracted_data.items():
        # if key is of the format "frictionN", then we need to convert the values to integers
        if re.match(r"friction(\d+)", key):
            # check if all items can be converted to integers and if so, convert them
            try : 
                extracted_data[key] = [int(item) for item in value]
            except ValueError:
                # if it's a single item, then it can be converted to an integer and we can move on 
                if len(value[0].split()) == 1: 
                    # discard this key and value 
                    to_remove_keys.append(key)
                    continue
                
                # if it's two items, then we need to extract the integer from the string
                extracted_data[key] = [int(item.split(" ")[1]) for item in value]

    for key in to_remove_keys:
        extracted_data.pop(key)

    return extracted_data



def get_macro_metrics(convo_ids: list[str],
                    model_outputs: Phase1OutputLoader, 
                    parsing_function: callable) -> dict: 
    
    overlap = 0 
    total_human_intervals = 0
    total_model_intervals = 0

    for convo_id in tqdm(convo_ids):
        #print(convo_id)

        # load the human annotations 
        human_annotations = loader.get_annotation_data(convo_id)
        friction_turns = human_annotations["friction_turns"]

        # if no friction turns are present, skip the conversation
        if len(friction_turns) == 0:
            #continue
            pass 

        human_response_intervals = [interval for order, interval in friction_turns.items()]

        
        model_response = model_outputs.get_output_of_convo(convo_id)
        model_response_dict = parsing_function(model_response["response"])

        if "friction_present" not in model_response_dict:
            print("Something has gone wrong")

        # if keys start with "friction", then it is a friction interval
        # make all intervals ints 
        model_response_intervals = [interval for order, interval in model_response_dict.items() if re.match(r"friction(\d+)", order)] 

        metrics = calculate_overlap(human_response_intervals, model_response_intervals)
        
        overlap += metrics["overlap"]
        total_human_intervals += metrics["total_human_intervals"]
        total_model_intervals += metrics["total_model_intervals"]

    results =  {
        "overlap": overlap,
        "total_human_intervals": total_human_intervals,
        "total_model_intervals": total_model_intervals,
    }

    precision = overlap / total_model_intervals if total_model_intervals > 0 else 0
    recall = overlap / total_human_intervals if total_human_intervals > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results["precision"] = round(precision*100, 4)
    results["recall"] = round(recall*100, 4)
    results["f1"] = round(f1*100, 4)

    return results

# load some of the model results 
# gpt4o = Phase1OutputLoader("/fs/clip-political/rupak/common_ground/experiments/phase1_experiments/phase1_outputs/friction_prediction_outputs/friction_detection_temp_0.01_gpt-4o_wo_gpt_assist.jsonl")
# gpt4omini = Phase1OutputLoader("/fs/clip-political/rupak/common_ground/experiments/phase1_experiments/phase1_outputs/friction_prediction_outputs/friction_detection_temp_0.01_gpt-4o-mini_wo_gpt_assist.jsonl")

# gpt4omini_w_assist = Phase1OutputLoader("/fs/clip-political/rupak/common_ground/experiments/phase1_experiments/phase1_outputs/friction_prediction_outputs/friction_detection_temp_0.01_gpt-4o-mini_w_gpt_assist.jsonl")
# gpt4o_w_assist = Phase1OutputLoader("/fs/clip-political/rupak/common_ground/experiments/phase1_experiments/phase1_outputs/friction_prediction_outputs/friction_detection_temp_0.01_gpt-4o_w_gpt_assist.jsonl") 

# llama8b = Phase1OutputLoader("/fs/clip-political/rupak/common_ground/experiments/phase1_experiments/phase1_outputs/friction_prediction_outputs/friction_detection_temp_0.01_Llama-3.1-8B-Instruct_wo_gpt_assist.jsonl")
# llama70b = Phase1OutputLoader("/fs/clip-political/rupak/common_ground/experiments/phase1_experiments/phase1_outputs/friction_prediction_outputs/friction_detection_temp_0.01_Llama-3.1-70B-Instruct_wo_gpt_assist.jsonl")

# llama8b_w_assist = Phase1OutputLoader("/fs/clip-political/rupak/common_ground/experiments/phase1_experiments/phase1_outputs/friction_prediction_outputs/friction_detection_temp_0.01_Llama-3.1-8B-Instruct_w_gpt_assist.jsonl")
# llama70b_w_assist = Phase1OutputLoader("/fs/clip-political/rupak/common_ground/experiments/phase1_experiments/phase1_outputs/friction_prediction_outputs/friction_detection_temp_0.01_Llama-3.1-70B-Instruct_w_gpt_assist.jsonl")