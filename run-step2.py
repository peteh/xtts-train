import torch 
import traceback
import os
import shutil
from utils.gpt_train import train_gpt
from config import TARGET_LANGUAGE, DATASET_PATH, TRAIN_BATCH_SIZE, TRAIN_EPOCHS, TRAIN_GRAD_ACCUMULATION_STEPS, TRAIN_MAX_PERMIT_AUDIO_S, TRAIN_OUTPUT_PATH, TRAIN_TRAINING_PATH

TRAIN_CSV = "metadata_train.csv"
EVAL_CSV = "metadata_eval.csv"
def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_model(language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, training_path, max_audio_length):
                clear_gpu_cache()

                # convert seconds to waveform frames
                max_audio_length = int(max_audio_length * 22050)
                config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=training_path, max_audio_length=max_audio_length)

                # copy original files to avoid parameters changes issues
                os.system(f"cp {config_path} {exp_path}")
                os.system(f"cp {vocab_file} {exp_path}")

                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
                print("Model training done!")
                clear_gpu_cache()
                return config_path, vocab_file, ft_xtts_checkpoint, speaker_wav

def finalize(output_dir, config_file, vocab_file, checkpoint_path, speaker_wav):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]
    model_dir = os.path.dirname(checkpoint_path)
    reduced_model_path = os.path.join(output_dir, "model.pth")
    torch.save(checkpoint, reduced_model_path)
    shutil.copy2(config_file, os.path.join(output_dir, "config.json"))
    shutil.copy2(vocab_file, os.path.join(output_dir, "vocab.json"))
    shutil.copy2(speaker_wav, os.path.join(output_dir, "reference.wav"))
    return reduced_model_path
    

config_path, vocab_file, ft_xtts_checkpoint, speaker_wav = train_model(TARGET_LANGUAGE, os.path.join(DATASET_PATH, TRAIN_CSV), os.path.join(DATASET_PATH, EVAL_CSV), TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_GRAD_ACCUMULATION_STEPS,  TRAIN_TRAINING_PATH, TRAIN_MAX_PERMIT_AUDIO_S)
finalize(TRAIN_OUTPUT_PATH, config_path, vocab_file, ft_xtts_checkpoint, speaker_wav)
print(f"config_path: {config_path}")
print(f"vocab_file: {vocab_file}")
print(f"ft_xtts_checkpoint: {ft_xtts_checkpoint}")
print(f"speaker_wav: {speaker_wav}")