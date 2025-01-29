Survival and recurrence prediction on breast cancer Whole Slide Images using attention based deep multiple instance learning, Cox proportional hazards model and self supervised feature extraction. 

Code References: CLAM (https://doi.org/10.1038/s41551-020-00682-w), RetCCL (https://doi.org/10.1016/j.media.2022.102645), Cox-AMIL (https://doi.org/10.48550/arXiv.2212.07724), Cox PH (https://doi.org/10.1111/j.2517-6161.1972.tb00899.x)

Pipeline:

#Please make sure to create an environment according to the "environment.yml" file. If you are on a windows machine, please see "wsi_core/WholeSlideImage.py", "extract_features_fp.py" and "vis_utils/heatmap_utils.py" scripts and change the necessary places as instructed.

cd CCL-CLAM-Cox

CREATE PATCHES:

#Create "DATA_DIRECTORY" and "RESULTS_DIRECTORY" folders. Put all of your images to 'DATA_DIRECTORY' folder and make sure the names of all the images are in the form 'slide_ID_COHORT.svs'. After you run the codes, 'masks', 'stitches', 'patches', 'process_list_autogen.csv' will be created inside the 'RESULTS_DIRECTORY' folder.

python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 

FEATURE EXTRACTION:

#Create "DIR_TO_COORDS" and "FEATURES_DIRECTORY" folders. From "RESULTS_DIRECTORY", copy "process_list_autogen.csv" to CCL-CLAM-Cox and "patches" folder to "DIR_TO_COORDS". Change "--premodel_type" argument from "imagenet" to "retccl" in order to implement RetCCL. After you run the codes, 'h5_files' and 'pt_files' folders will be created inside the 'FEATURES_DIRECTORY' folder.

python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path process_list_autogen.csv --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs --premodel_type retccl

TRAINING SPLITS:

#For survival prediction (regression): Create "dataset_csv" folder and inside, create "surv_pred.csv" according to the dataset (slide_id's should not contain .svs). While you are training and creating heatmaps for a specific model, make sure the name of the csv file is 'surv_pred.csv'. Change the name of the csv file only after you have trained the model and produced all of the heatmaps. When you are done with the previous model's heatmaps and you are training a different model, do not forget to change the name of the csv file to something other than 'surv_pred.csv', because only the csv file of the model you are working on should be called 'surv_pred.csv', otherwise the codes will take the wrong csv file as input. Create "splits" folder. After you run the codes, 'task_3_survival_prediction__100' folder will be created inside the 'splits' folder. While you are training and creating heatmaps for a specific model, make sure the name of the splits folder is 'task_3_survival_prediction__100'. Change the name of the splits folder only after you have trained the model and produced all of the heatmaps. When you are done with the previous model's heatmaps and you are training a different model, do not forget to change the name of the previous model's splits folder to something other than 'task_3_survival_prediction__100', because only the splits folder of the model you are working on should be called 'task_3_survival_prediction__100', otherwise the codes will take the wrong splits as input.

python create_splits_seq.py --task task_3_survival_prediction --seed 1 --k 1 --label_frac 1 --val_frac 0.2 --test_frac 0.2 --csv_path dataset_csv/surv_pred.csv

#For recurrence prediction (classification): Create "dataset_csv" folder and inside, create "recurr_pred.csv" according to the dataset (slide_id's should not contain .svs). While you are training and creating heatmaps for a specific model, make sure the name of the csv file is 'recurr_pred.csv'. Change the name of the csv file only after you have trained the model and produced all of the heatmaps. When you are done with the previous model's heatmaps and you are training a different model, do not forget to change the name of the csv file to something other than 'recurr_pred.csv', because only the csv file of the model you are working on should be called 'recurr_pred.csv', otherwise the codes will take the wrong csv file as input. Create "splits" folder. After you run the codes, 'recurrence_prediction__100' folder will be created inside the 'splits' folder. While you are training and creating heatmaps for a specific model, make sure the name of the splits folder is 'recurrence_prediction__100'. Change the name of the splits folder only after you have trained the model and produced all of the heatmaps. When you are done with the previous model's heatmaps and you are training a different model, do not forget to change the name of the previous model's splits folder to something other than 'recurrence_prediction__100', because only the splits folder of the model you are working on should be called 'recurrence_prediction__100', otherwise the codes will take the wrong splits as input.

python create_splits_seq.py --task recurrence_prediction --seed 1 --k 1 --label_frac 1 --val_frac 0.2 --test_frac 0.2 --csv_path dataset_csv/recurr_pred.csv

TRAINING:

#For survival prediction (regression): Create "DATA_ROOT_DIR" folder. Copy "FEATURES_DIRECTORY" folder to "DATA_ROOT_DIR". Change "model_size" argument from "small" to "retccl_comp" in order to use the features extracted by RetCCL. When you are training a model that does not contain some of the images, do not forget to exclude their features from the folders that contain the features. After you run the codes, 'results_surv' folder will be created in CCL-CLAM-Cox. When you are done training a specific model and you are about to create heatmaps for that specific model, make sure the name of the results folder is 'results_surv'. Change the name of the results folder only after you have produced all of the heatmaps. When you are done with the previous model's heatmaps and you are training a different model, do not forget to change the name of the previous model's results folder to something other than 'results_surv', because only the results folder of the model you are working on should be called 'results_surv', otherwise the codes will take the wrong trained model as input. 

python main_survival.py --drop_out --early_stopping --lr 2e-4 --k 1 --label_frac 1 --exp_code task_3_survival_prediction_clam --bag_loss ce --task task_3_survival_prediction --model_type amil --model_size retccl_comp --log_data --data_root_dir DATA_ROOT_DIR --csv_path dataset_csv/surv_pred.csv --feature_dir FEATURES_DIRECTORY --split_dir task_3_survival_prediction__100

#For recurrence prediction (classification): Create "DATA_ROOT_DIR" folder. Copy "pt_files" folder from "FEATURES_DIRECTORY" folder to "DATA_ROOT_DIR". Change "model_size" argument from "small" to "retccl_comp" in order to use the features extracted by RetCCL. When you are training a model that does not contain some of the images, do not forget to exclude their features from the folders that contain the features. After you run the codes, 'results_recurr' folder will be created in CCL-CLAM-Cox. When you are done training a specific model and you are about to create heatmaps for that specific model, make sure the name of the results folder is 'results_recurr'. Change the name of the results folder only after you have produced all of the heatmaps. When you are done with the previous model's heatmaps and you are training a different model, do not forget to change the name of the previous model's results folder to something other than 'results_recurr', because only the results folder of the model you are working on should be called 'results_recurr', otherwise the codes will take the wrong trained model as input. 

python main_recurrence.py --drop_out --early_stopping --lr 2e-4 --k 1 --label_frac 1 --exp_code recurrence_prediction_clam --bag_loss ce --inst_loss svm --task recurrence_prediction --model_type clam_mb --model_size retccl_comp --log_data --data_root_dir DATA_ROOT_DIR --csv_path dataset_csv/recurr_pred.csv --feature_dir FEATURES_DIRECTORY --split_dir recurrence_prediction__100 --subtyping

HEATMAP:

#For survival prediction (regression): Create "process_lists" folder inside "heatmaps" folder and copy "surv_pred.csv" from "dataset_csv" to there. In the "heatmaps/configs/config_template_surv.yml" file, change "model_size" argument from "small" to "retccl_comp" and change "premodel_type" argument from "imagenet" to "retccl" in order to use the RetCCL compatible model. Change "model_type" argument from "clam_mb" to "amil". After you run the codes, 'heatmap_production_results_surv', 'heatmap_raw_results_surv' folders will be created inside the 'heatmaps' folder. When you are done creating the heatmaps of a specific model and you are about to create heatmaps for a different model, do not forget to change the names of the previous 'heatmap_production_results_surv', 'heatmap_raw_results_surv' folders to something other than 'heatmap_production_results_surv', 'heatmap_raw_results_surv', otherwise the codes will overwrite those folders and everything inside. Also when you are about to create heatmaps for a different model, do not forget to replace the previous model's 'surv_pred.csv' csv file inside 'process_lists' folder with the csv file of the new model.

python create_heatmaps_survival.py --config config_template_surv.yaml

#For recurrence prediction (classification): Create "process_lists" folder inside "heatmaps" folder and copy "recurr_pred.csv" from "dataset_csv" to there. In the "heatmaps/configs/config_template_recurr.yml" file, change "model_size" argument from "small" to "retccl_comp" and change "premodel_type" argument from "imagenet" to "retccl" in order to use the RetCCL compatible model. After you run the codes, 'heatmap_production_results_recurr', 'heatmap_raw_results_recurr' folders will be created inside the 'heatmaps' folder. When you are done creating the heatmaps of a specific model and you are about to create heatmaps for a different model, do not forget to change the names of the previous 'heatmap_production_results_recurr', 'heatmap_raw_results_recurr' folders to something other than 'heatmap_production_results_recurr', 'heatmap_raw_results_recurr' otherwise the codes will overwrite those folders and everything inside. Also when you are about to create heatmaps for a different model, do not forget to replace the previous model's 'recurr_pred.csv' csv file inside 'process_lists' folder with the csv file of the new model.

python create_heatmaps_recurrence.py --config config_template_recurr.yaml

PREDICTED LABELS:

#Directly run the predicted_labels.py script. Make sure to read the instructions. For classification models only.

CONFUSION MATRIX:

#Directly run the confusion_matrix.py script. Make sure to read the instructions. For classification models only.

RISK SCORES:

#Directly run the risk_scores.py script. Make sure to read the instructions. For regression models only.
