from datasets import load_dataset
raw_dataset = load_dataset('super_glue','copa')
raw_dataset.save_to_disk('superglue/super_glue.copa')