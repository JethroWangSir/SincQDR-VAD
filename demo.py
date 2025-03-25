import os
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
import torchaudio
import torchaudio.transforms as T
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from model.tinyvad import TinyVAD

# Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Font configuration
font_path = '/share/nas169/jethrowang/fonts/Times_New_Roman.ttf'
font_prop = FontProperties(fname=font_path, size=18)

# Model and Processing Parameters
WINDOW_SIZE = 0.63
SINC_CONV = False
SSM = False
TARGET_SAMPLE_RATE = 16000

# Model Initialization
model = TinyVAD(1, 32, 64, patch_size=8, num_blocks=2, 
                sinc_conv=SINC_CONV, ssm=SSM).to(device)
checkpoint_path = '/share/nas169/jethrowang/SincVAD/exp/exp_0.63_tinyvad_psq_0.05/model_epoch_37_val_auroc=0.8894.ckpt'
model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
model.eval()

# Audio Processing Transforms
mel_spectrogram = T.MelSpectrogram(sample_rate=TARGET_SAMPLE_RATE, n_mels=64, win_length=400, hop_length=160)
log_mel_spectrogram = T.AmplitudeToDB()

# Chunking Parameters
chunk_duration = WINDOW_SIZE
shift_duration = WINDOW_SIZE * 0.875  # Increased overlap compared to first version

def predict(audio_record, audio_upload, threshold):
    """
    Predict voice activity in an audio file with detailed processing and visualization.
    
    Args:
        audio_file (str): Path to the audio file
        threshold (float): Decision threshold for speech/non-speech classification
    
    Yields:
        Intermediate and final prediction results
    """
    start_time = time.time()

    audio_input = audio_record if audio_record else audio_upload
    if not audio_input:
        return "No audio provided!", 0.0, "N/A", None

    try:
        # Load and preprocess audio
        waveform, orig_sample_rate = torchaudio.load(audio_input)
        
        # Resample if necessary
        if orig_sample_rate != TARGET_SAMPLE_RATE:
            print(f"Resampling from {orig_sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz")
            resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Ensure mono channel
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        yield "Error loading audio file.", None, None, None
        return
    
    # Audio duration checks and padding
    audio_duration = waveform.size(1) / TARGET_SAMPLE_RATE
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Original sample rate: {orig_sample_rate} Hz")
    print(f"Current sample rate: {TARGET_SAMPLE_RATE} Hz")

    if audio_duration < chunk_duration:
        required_length = int(chunk_duration * TARGET_SAMPLE_RATE)
        padding_length = required_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding_length))
    
    # Chunk processing parameters
    chunk_size = int(chunk_duration * TARGET_SAMPLE_RATE)
    shift_size = int(shift_duration * TARGET_SAMPLE_RATE)
    num_chunks = (waveform.size(1) - chunk_size) // shift_size + 1

    predictions = []
    time_stamps = []
    detailed_predictions = []

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlabel('Time (seconds)', fontproperties=font_prop)
    ax.set_ylabel('Probability', fontproperties=font_prop)
    ax.set_title('Voice Activity Detection Probability Over Time', fontproperties=font_prop)
    ax.axhline(y=threshold, color='tab:red', linestyle='--', label='Threshold')
    ax.grid(True)
    ax.set_ylim([-0.05, 1.05])

    # Process audio in chunks
    for i in range(num_chunks):
        start_idx = i * shift_size
        end_idx = start_idx + chunk_size
        chunk = waveform[:, start_idx:end_idx]

        if chunk.size(1) < chunk_size:
            break

        # Feature extraction
        inputs = mel_spectrogram(chunk)
        inputs = log_mel_spectrogram(inputs).to(device).unsqueeze(0)

        # Model inference
        with torch.no_grad():
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
        
        # Process outputs
        predictions.append(outputs.item())
        time_stamps.append(start_idx / TARGET_SAMPLE_RATE)
        
        detailed_predictions.append({
            'start_time': start_idx / TARGET_SAMPLE_RATE,
            'output': outputs.item(),
        })

        # Update plot dynamically
        ax.clear()
        ax.set_xlabel('Time (seconds)', fontproperties=font_prop)
        ax.set_ylabel('Probability', fontproperties=font_prop)
        ax.set_title('Speech Probability Over Time', fontproperties=font_prop)
        ax.axhline(y=threshold, color='tab:red', linestyle='--', label='Threshold')
        ax.grid(True)
        ax.set_ylim([-0.05, 1.05])
        ax.plot(time_stamps, predictions, label='Speech Probability', color='tab:blue')
        plt.tight_layout()

        # Yield intermediate progress
        yield "Processing...", None, None, fig

    # Detailed logging
    print("Detailed Predictions:")
    for pred in detailed_predictions:
        print(f"Start Time: {pred['start_time']:.2f}s, Output: {pred['output']:.4f}")

    # Final prediction processing
    avg_output = max(0, min(1, np.mean(predictions)))
    prediction_time = time.time() - start_time

    prediction = "Speech" if avg_output > threshold else "Non-speech"
    probability = f'{(float(avg_output) * 100):.2f}'
    inference_time = f'{prediction_time:.4f}'

    print(f"Final Prediction: {prediction}")
    print(f"Average Probability: {probability}%")
    print(f"Number of chunks processed: {num_chunks}")

    # Final result
    yield prediction, probability, inference_time, fig

# Gradio Interface
with gr.Blocks() as demo:
    gr.Image("./img/logo.png", elem_id="logo", height=100)
    # Title and Description
    gr.Markdown("<h1 style='text-align: center; color: black;'>Voice Activity Detection using SincVAD</h1>")
    gr.Markdown("<h3 style='text-align: center; color: black;'>Record or upload audio to predict speech activity and view the probability curve.</h3>")
    
    # Interface Layout
    with gr.Row():
        with gr.Column():
            # Separate recording and file upload
            record_input = gr.Microphone(type="filepath", label="Record Audio")
            upload_input = gr.Audio(type="filepath", label="Upload Audio")
            threshold_input = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="Threshold")
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction")
            probability_output = gr.Number(label="Average Probability (%)")
            time_output = gr.Textbox(label="Inference Time (seconds)")
        
    plot_output = gr.Plot(label="Probability Curve")

    # Prediction Trigger
    predict_btn = gr.Button("Start Prediction")
    predict_btn.click(
        predict, 
        [record_input, upload_input, threshold_input], 
        [prediction_output, probability_output, time_output, plot_output],
        api_name="predict"
    )


# Launch Configuration
if __name__ == "__main__":
    demo.queue()  # Enable queue to support generators
    demo.launch(share=True)
