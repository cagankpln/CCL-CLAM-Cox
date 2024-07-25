Survival and recurrence prediction on breast cancer Whole Slide Images using attention based deep multiple instance learning, Cox proportional hazards model and self supervised feature extraction. 

Code References: CLAM (https://doi.org/10.1038/s41551-020-00682-w), RetCCL (https://doi.org/10.1016/j.media.2022.102645), Cox-AMIL (https://doi.org/10.48550/arXiv.2212.07724), Cox PH (https://doi.org/10.1111/j.2517-6161.1972.tb00899.x)

Pipeline:

#NOTE: If you are on a windows machine, please see "wsi_core/WholeSlideImage.py", "extract_features_fp.py" and "vis_utils/heatmap_utils.py" scripts and change the necessary places as instructed. Also please make sure to create an environment according to the "environment.yml" file.

cd CCL-CLAM-Cox

CREATE PATCHES:

#Create "DATA_DIRECTORY" and "RESULTS_DIRECTORY" folders. Add "slide_" to the start of each data name in "DATA_DIRECTORY".

python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 

FEATURE EXTRACTION:

#Create "DIR_TO_COORDS" and "FEATURES_DIRECTORY" folders. From "RESULTS_DIRECTORY", copy "process_list_autogen.csv" to CCL-CLAM-Cox and "patches" folder to "DIR_TO_COORDS". Change "--premodel_type" argument from "imagenet" to "retccl" in order to implement RetCCL.

python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path process_list_autogen.csv --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs --premodel_type retccl

TRAINING SPLITS:

#For survival prediction: Create "dataset_csv" folder and inside, create "surv_pred.csv" according to the dataset (slide_id's should not contain .svs). Create "splits" folder.

python create_splits_seq.py --task task_3_survival_prediction --seed 1 --k 1 --label_frac 1 --val_frac 0.2 --test_frac 0.2 --csv_path dataset_csv/surv_pred.csv

#For recurrence prediction: Create "dataset_csv" folder and inside, create "recurr_pred.csv" according to the dataset (slide_id's should not contain .svs). Create "splits" folder.

python create_splits_seq.py --task recurrence_prediction --seed 1 --k 1 --label_frac 1 --val_frac 0.2 --test_frac 0.2 --csv_path dataset_csv/recurr_pred.csv

TRAINING:

#For survival prediction: Create "DATA_ROOT_DIR" folder. Copy "FEATURES_DIRECTORY" folder to "DATA_ROOT_DIR". Change "model_size" argument from "small" to "retccl_comp" in order to use the features extracted by RetCCL.

python main_survival.py --drop_out --early_stopping --lr 2e-4 --k 1 --label_frac 1 --exp_code task_3_survival_prediction_clam --bag_loss ce --task task_3_survival_prediction --model_type amil --model_size retccl_comp --log_data --data_root_dir DATA_ROOT_DIR --csv_path dataset_csv/surv_pred.csv --feature_dir FEATURES_DIRECTORY --split_dir task_3_survival_prediction__100

#For recurrence prediction: Create "DATA_ROOT_DIR" folder. Copy "pt_files" folder from "FEATURES_DIRECTORY" folder to "DATA_ROOT_DIR". Change "model_size" argument from "small" to "retccl_comp" in order to use the features extracted by RetCCL.

python main_recurrence.py --drop_out --early_stopping --lr 2e-4 --k 1 --label_frac 1 --exp_code recurrence_prediction_clam --bag_loss ce --inst_loss svm --task recurrence_prediction --model_type clam_mb --model_size retccl_comp --log_data --data_root_dir DATA_ROOT_DIR --csv_path dataset_csv/recurr_pred.csv --feature_dir FEATURES_DIRECTORY --split_dir recurrence_prediction__100 --subtyping

HEATMAP:

#For survival prediction: Create "process_lists" folder inside "heatmaps" folder and copy "surv_pred.csv" from "dataset_csv" to there. In the "heatmaps/configs/config_template_surv.yml" file, change "model_size" argument from "small" to "retccl_comp" and change "premodel_type" argument from "imagenet" to "retccl" in order to use the RetCCL compatible model. Change "model_type" argument from "clam_mb" to "amil".

python create_heatmaps_survival.py --config config_template_surv.yaml

#For recurrence prediction: Create "process_lists" folder inside "heatmaps" folder and copy "recurr_pred.csv" from "dataset_csv" to there. In the "heatmaps/configs/config_template_recurr.yml" file, change "model_size" argument from "small" to "retccl_comp" and change "premodel_type" argument from "imagenet" to "retccl" in order to use the RetCCL compatible model.

python create_heatmaps_recurrence.py --config config_template_recurr.yaml


