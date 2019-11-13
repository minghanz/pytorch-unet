for pre_weight in 'with_color_Mon Nov 11 22:24:57 2019'
do
    python run.py --run_eval --pretrained_weight_path='saved_models/'"$pre_weight"'/epoch00_6000.pth'
done