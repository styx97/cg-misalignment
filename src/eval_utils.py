import re 
from src.annotation_loader import Phase1AnnotationLoader


def extract_friction_details_corrected_new(text):
	# Initialize pattern to capture various parts of the input
	pattern = re.compile(
		r'friction_present:\s*(true|false)\s*|'          # Match "friction_present: true/false"
		r'"friction(\d+)":\s*\[([^\]]+)\]|'              # Match "frictionN": [x, y]
		r'"explanation(\d+)":\s*"((?:[^"]|\\")*?)"',     # Match "explanationN": including any " escaped with \"
		re.DOTALL
	)

	extracted_data = {}

	for match in pattern.finditer(text):
		if match.group(1) is not None:
			# Extracting the presence of friction
			extracted_data["friction_present"] = match.group(1).lower() == "true"
		elif match.group(2) is not None:
			# Extracting the friction index and values
			index = match.group(2)
			# Handle each item as a string rather than trying to convert to int immediately
			items = [x.strip().strip('"') for x in match.group(3).split(",")]
			extracted_data[f"friction{index}"] = items
		elif match.group(4) is not None:
			# Extracting explanations
			index = match.group(4)
			explanation = match.group(5).replace('\\"', '"')  # Correct escaped quotes
			extracted_data[f"explanation{index}"] = explanation

	# Post-process to convert the friction values to integers where applicable
	to_remove_keys = []
	for key, value in extracted_data.items():
		if re.match(r"friction(\d+)", key):
			# Try converting items to integers or extract from strings
			try:
				extracted_data[key] = [int(item) for item in value]
			except ValueError:
				# Handle cases where items are strings with "Turn X"
				if len(value[0].split()) == 1:
					# Discard this key and value if conversion is not applicable
					to_remove_keys.append(key)
					continue

				# Extract integer from strings like "Turn X"
				extracted_data[key] = [int(item.split(" ")[1]) for item in value]

	for key in to_remove_keys:
		extracted_data.pop(key)

	return extracted_data

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
					parsing_function: callable, 
					interval_overlap_function: callable) -> dict: 
	
	overlap = 0 
	total_human_intervals = 0
	total_model_intervals = 0

	for convo_id in convo_ids:
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

		metrics = interval_overlap_function(human_response_intervals, model_response_intervals)
		
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


def get_micro_metrics(convo_ids: list[str],
					model_outputs: Phase1OutputLoader, 
					parsing_function: callable, 
					interval_overlap_function: callable) -> dict: 
	
	precision = 0
	recall = 0
	f1 = 0

	for convo_id in convo_ids:
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

		metrics = interval_overlap_function(human_response_intervals, model_response_intervals)
		
		precision += metrics["precision"]
		recall += metrics["recall"]
		f1 += metrics["f1"]

	precision = precision / len(convo_ids)
	recall = recall / len(convo_ids)
	f1 = f1 / len(convo_ids)

	results = {}
	results["precision"] = round(precision*100, 4)
	results["recall"] = round(recall*100, 4)
	results["f1"] = round(f1*100, 4)

	return results



def compute_model_metrics(context, batch_test_map, model_name="distilroberta"):
    """
    Compute metrics for model outputs given a context value.
    
    Args:
        context (int): Context value for the model
        batch_test_map (dict): Mapping of batch indices to conversation ID lists
        model_name (str): Name of the model (default: "distilroberta")
    
    Returns:
        dict: Dictionary containing mean values for all metrics
    """
    import pandas as pd
    import numpy as np
    from src.utils import read_jsonlines
    
    # Initialize metric lists
    overlap_precisions = []
    overlap_recalls = []
    overlap_f1s = []
    
    span_overlap_precisions = []
    span_overlap_recalls = [] 
    span_overlap_f1s = []
    
    for index, convo_id_list in enumerate(batch_test_map.values()):
        path_to_data_file = f"../data/model_outputs/{model_name}_outputs/context_{context}/split_{index}/test.csv"
        path_to_predictions = f"../data/model_outputs/{model_name}_outputs/context_{context}/split_{index}/predict_results_None.txt"
        
        df = pd.read_csv(path_to_data_file)
        predictions = read_jsonlines(path_to_predictions)
        preds = [p["prediction"] for p in predictions]
        df["prediction"] = preds

        model_output_dict = {}

        for conv_id in convo_id_list: 
            intervals = find_intervals_by_conv_id(conv_id, df)
            model_output_dict[conv_id] = intervals

        print(f"Batch: {index+1}")
        for metric_function in [get_macro_metrics_new]:
            print(f"Metric Function: {metric_function.__name__}")
            for interval_matching_function in [calculate_overlap, calculate_span_overlap]:
                print(f"Interval Matching Function: {interval_matching_function.__name__}")
                results = metric_function(convo_id_list, model_output_dict, interval_matching_function)
                precision, recall, f1 = results["precision"], results["recall"], results["f1"]
                #print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

                if interval_matching_function == calculate_overlap:
                    overlap_precisions.append(precision)
                    overlap_recalls.append(recall)
                    overlap_f1s.append(f1)

                elif interval_matching_function == calculate_span_overlap:
                    span_overlap_precisions.append(precision)
                    span_overlap_recalls.append(recall)
                    span_overlap_f1s.append(f1)
        
                #print("\n")
            print("\n")
    
    # Compute means
    results = {
        "overlap_precision_mean": np.mean(overlap_precisions),
        "overlap_recall_mean": np.mean(overlap_recalls),
        "overlap_f1_mean": np.mean(overlap_f1s),
        "span_overlap_precision_mean": np.mean(span_overlap_precisions),
        "span_overlap_recall_mean": np.mean(span_overlap_recalls),
        "span_overlap_f1_mean": np.mean(span_overlap_f1s)
    }
    
    return results