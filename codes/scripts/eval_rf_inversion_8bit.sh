python3 rf_inversion_8bit.py \
	--model_path /data/chx/FLUX.1-dev \
	--eval_dataset EditEval_v1 \
	--height 512 \
	--width 512 \
	--dtype bfloat16 \
	--quant_8bit \
	--num_inversion_steps 28 \
	--num_inference_steps 28 \
	--output_dir outputs/rf_inversion_8bit \
	--save_samples

