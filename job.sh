#!/bin/bash
#SBATCH -J zhanghansheng_amphion
#SBATCH -A T00120230002
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH -o /mntcephfs/lab_data/zhangxueyao/zhanghansheng/Amphion/output/job.%j.out 
module load openmpi/gcc/64/4.1.2
source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate /mntnfs/lee_data1/zhanghansheng/anaconda3/envs/amphion_zhs/
cd /mntcephfs/lab_data/zhangxueyao/zhanghansheng/Amphion
sh egs/tts/Jets/run.sh --stage 2 --name rename_test --gpu "0"
# sh egs/tts/Jets/run.sh --stage 3 --gpu "0" \
#     --infer_expt_dir ckpts/tts/500_test \
#     --infer_output_dir ckpts/tts/500_test/result \
#     --infer_mode "batch" \
#     --infer_dataset "LJSpeech" \
#     --infer_testing_set "test" \
#     --vocoder_dir ckpts/vocoder/hifigan_ljspeech/checkpoints