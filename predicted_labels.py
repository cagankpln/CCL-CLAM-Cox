#THIS CODE IS ONLY FOR THE CLASSIFICATION MODELS!

import pandas as pd
import os
import numpy as np
import torch
from models.model_clam import CLAM_MB, CLAM_SB
from utils.eval_utils import initiate_model as initiate_model
import argparse

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Model Configuration")
parser.add_argument('--drop_out', type=bool, default=True, help='Enable dropout')
parser.add_argument('--model_type', type=str, default='clam_mb', help='Type of model to use')
parser.add_argument('--model_size', type=str, default='retccl_comp', help='Size of the model')
parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')
args = parser.parse_args()

#choose the model.
model =  initiate_model(args, 'results_recurr_nocutoff/recurrence_prediction_clam_s1/s_0_checkpoint.pt')

#choose the csv file.
df = pd.read_csv('dataset_csv/recurr_pred_nocutoff.csv')
ids = df['slide_id']

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB)):
			model_results_dict = model(features)
			logits, Y_prob, Y_hat, A, _ = model(features)
			Y_hat = Y_hat.item()

			if isinstance(model, (CLAM_MB,)):
				A = A[Y_hat]

			A = A.view(-1, 1).cpu().numpy()

		else:
			raise NotImplementedError

		print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
		
		probs, ids = torch.topk(Y_prob, k)
		probs = probs[-1].cpu().numpy()
		ids = ids[-1].cpu().numpy()
		preds_str = np.array([reverse_label_dict[idx] for idx in ids])

	return ids, preds_str, probs, A

label_dict = {'low':0, 'high':1}
class_labels = list(label_dict.keys())
class_encodings = list(label_dict.values())
reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

#when you are working on a model that does not contain some of the images, do not forget to exclude their features from the folders that contain the features.
for i in range(len(ids)):
    slide_name = df.loc[i, 'slide_id']
	#make sure the features_path points to the pt_files.
    features_path = os.path.join('DATA_ROOT_DIR/FEATURES_DIRECTORY/pt_files', slide_name + '.pt')
    features = torch.load(features_path)
    Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(
        model, 
        features, 
        df.loc[i, 'label'], 
        reverse_label_dict, 
        args.n_classes
    )
    
    for c in range(args.n_classes):
        df.loc[i, 'prediction_{}'.format(c)] = Y_hats_str[c]
        df.loc[i, 'probability_{}'.format(c)] = Y_probs[c]

#choose name of the output csv file.		
df.to_csv('predicted_labels_nocutoff.csv', index=False)