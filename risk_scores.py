#THIS CODE IS ONLY FOR THE REGRESSION MODELS!

import pandas as pd
import os
import torch
from models.model_amil import AMIL
from utils.eval_utils_survival import initiate_model as initiate_model

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

settings = {
			'drop_out': True,
			'model_type': 'amil',
			'model_size': 'retccl_comp',
			'print_model_info': False
			}

#choose the model.
model =  initiate_model(settings, 'results_surv_full/task_3_survival_prediction_clam_s1/s_0_checkpoint.pt')

#choose the csv file.
df = pd.read_csv('dataset_csv/surv_pred_full.csv')
ids = df['slide_id']

def infer_single_slide(model, features):
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, AMIL):
			risk, A, _= model(features)

			risk = risk.item()
			A = A.view(-1, 1).cpu().numpy()

		else:
			raise NotImplementedError

	return risk, A

for i in range(len(ids)):
    slide_name = df.loc[i, 'slide_id']
	#make sure the features_path points to the pt_files.
    features_path = os.path.join('DATA_ROOT_DIR/FEATURES_DIRECTORY/pt_files', slide_name+'.pt')
    features = torch.load(features_path)
    risk, A = infer_single_slide(model, features)
    df.loc[i, 'risk'] = risk

#choose name of the output csv file.	
df.to_csv('risk_scores_full.csv', index=False)