cd /mnt/sh/mmvision/home/jonahli/projects/SpatialVL/spatial-llm-rl/examples/grpo_trainer

python preprocess_geoloc_data.py \
    --input_file /mnt/sh/mmvision/home/jonahli/data/global-streetscapes/manual_labels/train/merged_labels_with_city.csv \
    --output_dir /mnt/sh/mmvision/home/jonahli/data_rl/global-streetscapes \
    --lat_column lat \
    --lon_column lon \
    --image_column img_path \
    --context_column None