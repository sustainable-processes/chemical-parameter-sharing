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
	python -m ps_model.gen_fp --clean_data_folder_path="data/mid_class_train.parquet" --fp_size=2048 --overwrite=False
fp_mid_class_test:
	python -m ps_model.gen_fp --clean_data_folder_path="data/mid_class_test.parquet" --fp_size=2048 --overwrite=False
fp_super_cc_class_train:
	python -m ps_model.gen_fp --clean_data_folder_path="data/super_class_cc_train.parquet" --fp_size=2048 --overwrite=False
fp_super_cc_class_test:
	python -m ps_model.gen_fp --clean_data_folder_path="data/super_class_cc_test.parquet" --fp_size=2048 --overwrite=False
fp_super_class_fgi_train:
	python -m ps_model.gen_fp --clean_data_folder_path="data/super_class_fgi_train.parquet" --fp_size=2048 --overwrite=False
fp_super_class_fgi_test:
	python -m ps_model.gen_fp --clean_data_folder_path="data/super_class_fgi_test.parquet" --fp_size=2048 --overwrite=False
fp_super_class_reductions_train:
	python -m ps_model.gen_fp --clean_data_folder_path="data/super_class_reductions_train.parquet" --fp_size=2048 --overwrite=False
fp_super_class_reductions_test:
	python -m ps_model.gen_fp --clean_data_folder_path="data/super_class_reductions_test.parquet" --fp_size=2048 --overwrite=False
fp_random_train:
	python -m ps_model.gen_fp --clean_data_folder_path="data/random_train.parquet" --fp_size=2048 --overwrite=False
fp_random_test:
	python -m ps_model.gen_fp --clean_data_folder_path="data/random_test.parquet" --fp_size=2048 --overwrite=False

gen_fp_1: fp_mid_class_train fp_mid_class_test fp_super_cc_class_train fp_super_cc_class_test

gen_fp_2: fp_super_class_fgi_train fp_super_class_fgi_test fp_super_class_reductions_train fp_super_class_reductions_test

gen_fp_random: fp_random_train fp_random_test

# 2. Evaluate models on random split of dataset

gao_random_split:
	python -m gao_model --model_type="gao_model" --train_data_path="data/random_train.parquet" --test_data_path="data/random_test.parquet" --output_folder_path="models/gao_random_split"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=50 --evaluate_on_test_data=True --early_stopping_patience=10 --wandb_entity=$(WANDB_ENTITY)

gao_random_split_OHE:
	python -m gao_model --model_type="gao_model" --train_data_path="data/random_test_fp_ohe.parquet" --test_data_path="data/random_test_fp_ohe.parquet" --output_folder_path="models/gao_ohe_random_split"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=50 --evaluate_on_test_data=True --early_stopping_patience=10 --wandb_entity=$(WANDB_ENTITY)

ps_random_split:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/random_train.parquet" --test_data_path="data/random_test.parquet" --output_folder_path="models/ps_random_split"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=50 --evaluate_on_test_data=True --early_stopping_patience=10 --wandb_entity=$(WANDB_ENTITY)


initial_random_split: gao_random_split ps_random_split




# 2.b Train & evaluate a model on each dataset
super_class_training_cc:
	python -m gao_model --model_type="gao_model" --train_data_path="data/super_class_cc_train.parquet" --test_data_path="data/super_class_cc_test.parquet" --output_folder_path="models/super_class_cc"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

super_class_fgi_training:
	python -m gao_model --model_type="gao_model" --train_data_path="data/super_class_fgi_train.parquet" --test_data_path="data/super_class_fgi_test.parquet" --output_folder_path="models/super_class_fgi"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

super_class_reductions_training:
	python -m gao_model --model_type="gao_model" --train_data_path="data/super_class_reductions_train.parquet" --test_data_path="data/super_class_reductions_test.parquet" --output_folder_path="models/super_class_reductions"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)
	
mid_class_training:
	python -m gao_model --model_type="gao_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/gao_mid_attempt2"  --train_fraction=1 --train_val_split=0.8 --overwrite=True --epochs=50 --evaluate_on_test_data=True --early_stopping_patience=10 --wandb_entity=$(WANDB_ENTITY)

# Generic model training
upstream_model_training:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream"  --train_fraction=1 --train_val_split=0.8 --overwrite=True --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

# Training fraction
upstream_20:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=True --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

upstream_40:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream_40"  --train_fraction=0.4 --train_val_split=0.8 --overwrite=True --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

upstream_60:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream_60"  --train_fraction=0.6 --train_val_split=0.8 --overwrite=True --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

upstream_80:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream_80"  --train_fraction=0.8 --train_val_split=0.8 --overwrite=True --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=100 --wandb_entity=$(WANDB_ENTITY)

# 3. Size of parameter sharing layer

upstream_05k:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream_05k"  --train_fraction=1 --train_val_split=0.8 --overwrite=True --epochs=50 --evaluate_on_test_data=True --early_stopping_patience=10 --wandb_entity=$(WANDB_ENTITY) --param_sharing_size=500

upstream_2k:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream_2k"  --train_fraction=1 --train_val_split=0.8 --overwrite=True --epochs=50 --evaluate_on_test_data=True --early_stopping_patience=10 --wandb_entity=$(WANDB_ENTITY) --param_sharing_size=2000

upstream_4k:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream_4k"  --train_fraction=1 --train_val_split=0.8 --overwrite=True --epochs=50 --evaluate_on_test_data=True --early_stopping_patience=10 --wandb_entity=$(WANDB_ENTITY) --param_sharing_size=4000

upstream_8k:
	python -m ps_model --model_type="upstream_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/upstream_8k"  --train_fraction=1 --train_val_split=0.8 --overwrite=True --epochs=50 --evaluate_on_test_data=True --early_stopping_patience=10 --wandb_entity=$(WANDB_ENTITY) --param_sharing_size=8000

upstream_ps_size: upstream_05k upstream_2k upstream_4k upstream_8k

# 4. Multi-ps-model
multi_ps_training:
	python -m ps_model --model_type="multi_ps_model" --train_data_path="data/mid_class_train.parquet" --test_data_path="data/mid_class_test.parquet" --output_folder_path="models/multi_ps"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=100 --evaluate_on_test_data=True --early_stopping_patience=10 --wandb_entity=$(WANDB_ENTITY)

