current_dir = $(shell pwd)
uid = $(shell id -u)
gid = $(shell id -g)
download_path=ord/

clean_default_num_agent=3
clean_default_num_cat=1
clean_default_num_reag=2
WANDB_ENTITY=ceb-sre
dataset_version=v5

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
	python -m condition_prediction.gen_fp --clean_data_folder_path="data/mid_class_train.parquet" --fp_size=2048 --overwrite=False
fp_mid_class_test:
	python -m condition_prediction.gen_fp --clean_data_folder_path="data/mid_class_test.parquet" --fp_size=2048 --overwrite=False
fp_super_cc_class_train:
	python -m condition_prediction.gen_fp --clean_data_folder_path="data/super_class_cc_train.parquet" --fp_size=2048 --overwrite=False
fp_super_cc_class_test:
	python -m condition_prediction.gen_fp --clean_data_folder_path="data/super_class_cc_test.parquet" --fp_size=2048 --overwrite=False
fp_super_class_fgi_train:
	python -m condition_prediction.gen_fp --clean_data_folder_path="data/super_class_fgi_train.parquet" --fp_size=2048 --overwrite=False
fp_super_class_fgi_test:
	python -m condition_prediction.gen_fp --clean_data_folder_path="data/super_class_fgi_test.parquet" --fp_size=2048 --overwrite=False
fp_super_class_reductions_train:
	python -m condition_prediction.gen_fp --clean_data_folder_path="data/super_class_reductions_train.parquet" --fp_size=2048 --overwrite=False
fp_super_class_reductions_test:
	python -m condition_prediction.gen_fp --clean_data_folder_path="data/super_class_reductions_test.parquet" --fp_size=2048 --overwrite=False

gen_fp_1: fp_mid_class_train fp_mid_class_test fp_super_cc_class_train fp_super_cc_class_test

gen_fp_2: fp_super_class_fgi_train fp_super_class_fgi_test fp_super_class_reductions_train fp_super_class_reductions_test

# 2. Train & evaluate a model on each dataset
super_class_training_cc:
	python -m condition_prediction --model_type="gao_model" --train_data_path="data/super_class_cc_train.parquet" --test_data_path="data/super_class_cc_test.parquet" --output_folder_path="models/super_class_cc"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

super_class_fgi_training:
	python -m condition_prediction --model_type="gao_model" --train_data_path="data/super_class_fgi_train.parquet" --test_data_path="data/super_class_fgi_test.parquet" --output_folder_path="models/super_class_fgi"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

super_class_reductions_training:
	python -m condition_prediction --model_type="gao_model" --train_data_path="data/super_class_reductions_train.parquet" --test_data_path="data/super_class_reductions_test.parquet" --output_folder_path="models/super_class_reductions"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)
	
mid_class_training:
	python -m condition_prediction --model_type="gao_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/mid_class"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

upstream_model_training:
	python -m condition_prediction --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

####################################################################################################
# 									make commands copied from ORDerly paper
####################################################################################################



# 1. gen fp

fp_no_trust_no_map_test:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_test.parquet" --fp_size=2048 --overwrite=False
fp_no_trust_no_map_train:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_train.parquet" --fp_size=2048 --overwrite=False

fp_no_trust_with_map_test:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_test.parquet" --fp_size=2048 --overwrite=False
fp_no_trust_with_map_train:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_train.parquet" --fp_size=2048 --overwrite=False

fp_with_trust_with_map_test:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_test.parquet" --fp_size=2048 --overwrite=False
fp_with_trust_with_map_train:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_train.parquet" --fp_size=2048 --overwrite=False

fp_with_trust_no_map_test:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_test.parquet" --fp_size=2048 --overwrite=False
fp_with_trust_no_map_train:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_train.parquet" --fp_size=2048 --overwrite=False

paper_8: fp_no_trust_no_map_test fp_no_trust_no_map_train fp_no_trust_with_map_test fp_no_trust_with_map_train fp_with_trust_with_map_test fp_with_trust_with_map_train fp_with_trust_no_map_test fp_with_trust_no_map_train

# 2. train models
#Remember to switch env here (must contain TF, e.g. tf_mac_m1)
# Full dataset
no_trust_no_map_train:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_test.parquet" --output_folder_path="models/no_trust_no_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

no_trust_with_map_train:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_test.parquet" --output_folder_path="models/no_trust_with_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

with_trust_no_map_train:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_test.parquet" --output_folder_path="models/with_trust_no_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

with_trust_with_map_train:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_test.parquet" --output_folder_path="models/with_trust_with_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

# 20% of data
no_trust_no_map_train_20:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_test.parquet" --output_folder_path="models/no_trust_no_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

no_trust_with_map_train_20:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_test.parquet" --output_folder_path="models/no_trust_with_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

with_trust_no_map_train_20:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_test.parquet" --output_folder_path="models/with_trust_no_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

with_trust_with_map_train_20:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_test.parquet" --output_folder_path="models/with_trust_with_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY)


# Sweeps
RANDOM_SEEDS = 12345 54321 98765
TRAIN_FRACS =   1.0 0.2 0.4 0.6 0.8
# Path on lightning
# DATASETS_PATH = /project/studios/orderly-preprocessing/ORDerly/data/orderly/datasets_$(dataset_version)/
# Normal path
DATASETS_PATH = ORDerly/data/orderly/datasets_$(dataset_version)/
DATASETS = no_trust_with_map  no_trust_no_map with_trust_with_map with_trust_no_map 
dataset_size_sweep:
	@for random_seed in ${RANDOM_SEEDS}; \
	do \
		for dataset in ${DATASETS}; \
		do \
			for train_frac in ${TRAIN_FRACS}; \
			do \
				rm -rf .tf_cache* && python -m condition_prediction --train_data_path=${DATASETS_PATH}/orderly_$${dataset}_train.parquet --test_data_path=${DATASETS_PATH}/orderly_$${dataset}_test.parquet --output_folder_path=models/$${dataset} --dataset_version=$(datset_version) --train_fraction=$${train_frac} --train_val_split=0.8 --random_seed=$${random_seed} --overwrite=True --batch_size=512 --epochs=100 --train_mode=0 --early_stopping_patience=0  --evaluate_on_test_data=True --wandb_entity=$(WANDB_ENTITY) ; \
			done \
		done \
	done


sweep_no_trust_no_map_train_commands:
	python -m sweep sweeps/no_trust_no_map_train.yaml --dry_run

sweep_no_trust_with_map_train_commands:
	python -m sweep sweeps/no_trust_with_map_train.yaml --dry_run

sweep_with_trust_no_map_train_commands:
	python -m sweep sweeps/with_trust_no_map_train.yaml --dry_run

sweep_with_trust_with_map_train_commands:
	python -m sweep sweeps/with_trust_with_map_train.yaml --dry_run

sweep_all: sweep_no_trust_no_map_train_commands sweep_no_trust_with_map_train_commands sweep_with_trust_no_map_train_commands sweep_with_trust_with_map_train_commands

train_all: no_trust_no_map_train no_trust_with_map_train with_trust_no_map_train with_trust_with_map_train no_trust_no_map_train_20 no_trust_with_map_train_20 with_trust_no_map_train_20 with_trust_with_map_train_20


