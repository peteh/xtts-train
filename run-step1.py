from utils.formatter import format_audio_list, list_audios
from config import AUDIO_PATH, TARGET_LANGUAGE, DATASET_PATH, WHISPER_MODEL, WHISPER_FORCE_CPU, WHISPER_THREADS

audio_files = list_audios(AUDIO_PATH)
train_meta, eval_meta, audio_total_size = format_audio_list(audio_files, 
                                                            target_language=TARGET_LANGUAGE, 
                                                            out_path=DATASET_PATH, 
                                                            gradio_progress=None, 
                                                            whisper_model=WHISPER_MODEL, 
                                                            whisper_threads = WHISPER_THREADS, 
                                                            force_cpu=WHISPER_FORCE_CPU)