for pre_weight in 'with_color_Sat Nov 16 00:03:56 2019' #'with_color_Sat Nov 16 00:02:24 2019' #'with_color_Mon Nov 11 22:24:57 2019'
do
    python run.py --run_eval --pretrained_weight_path='saved_models/'"$pre_weight"'/epoch00_20000.pth'
done