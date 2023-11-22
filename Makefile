current_dir = $(shell pwd)
uid = $(shell id -u)
gid = $(shell id -g)


clean_default_num_agent=3
clean_default_num_cat=1
clean_default_num_reag=2
WANDB_ENTITY=ceb-sre

####################################################################################################
# 									make commands for parameter sharing
####################################################################################################

# Start by finding orderly_condition_train.parquet and orderly_condition_test.parquet which are the files containing all the reactions.
# Then use gen_smi_files.ipynb to generate the smi files 
# Then use NameRxn by nextmove software to generate orderly_cond_classes.smi
# Then run the code in clustering.ipynb to generate the new datasets with different clusterings based on reaction classes
# Then use the make commands below

# 1. Generate fingerprints for each dataset
## Super class
fp_mid_class_train:
	python -m gao_model.gen_fp --clean_data_folder_path="data/mid_class_train.parquet" --fp_size=2048 --overwrite=False
fp_mid_class_test:
	python -m gao_model.gen_fp --clean_data_folder_path="data/mid_class_test.parquet" --fp_size=2048 --overwrite=False
fp_super_cc_class_train:
	python -m gao_model.gen_fp --clean_data_folder_path="data/super_class_cc_train.parquet" --fp_size=2048 --overwrite=False
fp_super_cc_class_test:
	python -m gao_model.gen_fp --clean_data_folder_path="data/super_class_cc_test.parquet" --fp_size=2048 --overwrite=False
fp_super_class_fgi_train:
	python -m gao_model.gen_fp --clean_data_folder_path="data/super_class_fgi_train.parquet" --fp_size=2048 --overwrite=False
fp_super_class_fgi_test:
	python -m gao_model.gen_fp --clean_data_folder_path="data/super_class_fgi_test.parquet" --fp_size=2048 --overwrite=False
fp_super_class_reductions_train:
	python -m gao_model.gen_fp --clean_data_folder_path="data/super_class_reductions_train.parquet" --fp_size=2048 --overwrite=False
fp_super_class_reductions_test:
	python -m gao_model.gen_fp --clean_data_folder_path="data/super_class_reductions_test.parquet" --fp_size=2048 --overwrite=False

gen_fp_1: fp_mid_class_train fp_mid_class_test fp_super_cc_class_train fp_super_cc_class_test

gen_fp_2: fp_super_class_fgi_train fp_super_class_fgi_test fp_super_class_reductions_train fp_super_class_reductions_test

# 2. Train & evaluate a model on each dataset
super_class_training_cc:
	python -m gao_model --model_type="gao_model" --train_data_path="data/super_class_cc_train.parquet" --test_data_path="data/super_class_cc_test.parquet" --output_folder_path="models/super_class_cc"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

super_class_fgi_training:
	python -m gao_model --model_type="gao_model" --train_data_path="data/super_class_fgi_train.parquet" --test_data_path="data/super_class_fgi_test.parquet" --output_folder_path="models/super_class_fgi"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

super_class_reductions_training:
	python -m gao_model --model_type="gao_model" --train_data_path="data/super_class_reductions_train.parquet" --test_data_path="data/super_class_reductions_test.parquet" --output_folder_path="models/super_class_reductions"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)
	
mid_class_training:
	python -m gao_model --model_type="gao_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/mid_class"  --train_fraction=0.01 --train_val_split=0.8 --overwrite=True --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

upstream_model_training:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream"  --train_fraction=0.01 --train_val_split=0.8 --overwrite=True --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)


