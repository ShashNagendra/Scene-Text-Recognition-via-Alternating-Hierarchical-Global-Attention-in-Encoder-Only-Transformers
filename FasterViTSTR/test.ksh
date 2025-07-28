CUDA_VISIBLE_DEVICES=0 python3 test.py --eval_data data_lmdb_release/evaluation --benchmark_all_eval --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=faster_vit_0_224 --sensitive --data_filtering_off  --imgH 224 --imgW 224 --saved_model <path/to/best_accuracy.pth>


CUDA_VISIBLE_DEVICES=0 python3 test.py --eval_data data_lmdb_release/evaluation --benchmark_all_eval --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=faster_vit_1_224 --sensitive --data_filtering_off  --imgH 224 --imgW 224 --saved_model <path/to/best_accuracy.pth>



CUDA_VISIBLE_DEVICES=0 python3 test.py --eval_data data_lmdb_release/evaluation --benchmark_all_eval --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=faster_vit_2_224 --sensitive --data_filtering_off  --imgH 224 --imgW 224 --saved_model <path/to/best_accuracy.pth>


CUDA_VISIBLE_DEVICES=0 python3 test.py --eval_data data_lmdb_release/evaluation --benchmark_all_eval --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=faster_vit_3_224 --sensitive --data_filtering_off  --imgH 224 --imgW 224 --saved_model <path/to/best_accuracy.pth>
