source ~/.bashrc_udacity
source deactivate
source activate carnd-term1-gpu
data_dir='/esat/diamond/liqianma/datasets/Udacity/Simulator-track1/data'
log_dir='/esat/diamond/liqianma/exp_logs/Udacity'


# source activate carnd-term1-gpu
# data_dir='Path_to_data_dir'
# log_dir='Path_to_log_dir'


model_dir=${log_dir}'/behavioral_clone_nvidia_model_deeper' # model=3
# ckpt_path=${model_dir}'/15.h5'
python model.py \
    --data_dir=${data_dir} \
    --model_dir=${model_dir} \
    --model=3 \
    --batch_size=128  --epoch=40 \
    # --ckpt_path=${ckpt_path}