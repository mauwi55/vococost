#!/usr/bin/env python3

import os
import streamlit as st
import librosa
from librosa.filters import get_window as get_window_librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from scipy.io import wavfile
from pathlib import Path
import gc
import tempfile
import io

# Config
st.set_page_config(
    page_title="Vococo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron&display=swap');
    
    .stApp {
        background: url('https://otasa.nekoweb.org/img/vococo_bg.webp') no-repeat center center fixed;
        background-size: cover;
    }
    
    .main > div {
        background: rgba(30, 41, 59, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    
    h1 {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
    }
    
    .stButton>button {
        opacity: 0.9;
        font-weight: bold;
    }
    
    .stExpander {
        background: rgb(23 37 49 / 74%);
        border-radius: 5px;
        mix-blend-mode: luminosity;
    }
    
    .audio-preview {
        background: rgba(255, 255, 255, 0.75);
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    
    .stAppViewContainer{
        background: rgba(10, 10, 10, 0.3);
    }
    .stExpanderDetails{
    }
    .st-emotion-cache-fis6aj{
    	background-color: rgba(100, 100, 100, 0.2);
    }
    .st-emotion-cache-1krtkoa{
    	background-color:rgb(91 12 12 / 80%);
    	border:1px solid rgb(91 12 12 / 80%);
    }
	.st-emotion-cache-1krtkoa:hover, .st-emotion-cache-1krtkoa:focus-visible {
    	background-color:rgb(91 12 12 / 100%);
    	border:1px solid rgb(91 12 12 / 100%);
    	mix-blend-mode: hard-light;
	}
	/*
	.st-dr{
    	background-color:rgb(255 255 255 / 100%);
	}
	*/
	.st-bq{
		background-color:rgb(1 10 88 / 62%);
	}
	div[role*="progressbar"] > div{
    	background-color:rgb(255 255 255 / 100%) !important;
	}
	div[role*="progressbar"] > div > div{
    	background-color:rgb(10 10 10 / 100%) !important;
	}
	div[role*="progressbar"] > div > div > div{
    	background-color:rgb(255 255 255 / 100%) !important;
	}

	ul[data-testid*="stSelectboxVirtualDropdown"]{
		background:none;
	}
	
	.stTooltipContent{
		color:#cccccc;
	}
	.stAlertContainer{
	    /*background-color: rgb(111 105 64 / 53%);*/
        background-color: rgb(127 141 70);
	    mix-blend-mode: color-burn;
	    color: white;
	}
    
</style>
""", unsafe_allow_html=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('grayscale')
    plt.rcParams['figure.max_open_warning'] = 0
except ImportError:
    st.error("Matplotlib required")
    plt = None

# Disable librosa cache
os.environ['LIBROSA_CACHE_DIR'] = tempfile.gettempdir()
os.environ['LIBROSA_CACHE_LEVEL'] = '10'

# ------------------------- Window Functions ---------------------------
def vorbis_window(N):
    x = np.arange(N, dtype=np.float32)
    return np.sin(0.5 * np.pi * (np.sin(np.pi * (x + 0.5) / N))**2)

def welch_window(N):
    x = np.arange(N, dtype=np.float32)
    center = (N - 1) / 2
    if center == 0: return np.ones(N, dtype=np.float32)
    return 1 - ((x - center) / center)**2

def kbd_window(N, beta=14.0):
    from scipy.special import i0  # Modified Bessel function of the first kind
    
    N = int(N)
    M = N // 2
    n = np.arange(0, M + 1)
    kaiser = i0(beta * np.sqrt(1 - (n / M) ** 2)) / i0(beta)
    
    kaiser_sum = np.cumsum(kaiser)
    kbd_half = np.sqrt(kaiser_sum[:-1] / kaiser_sum[-1])
    
    if N % 2 == 0:
        kbd = np.concatenate([kbd_half, kbd_half[::-1]])
    else:
        kbd = np.concatenate([kbd_half, [1.0], kbd_half[::-1]])
    
    return kbd.astype(np.float32)

def generate_window_spec(win_name, beta, alpha):
    if win_name == 'vorbis': return vorbis_window
    if win_name == 'welch': return welch_window
    elif win_name == 'kbd': return ('kbd', beta)
    elif win_name == 'kaiser': return ('kaiser', beta)
    elif win_name == 'tukey': return (win_name, alpha)
    else: return win_name

def generate_window_array(win_spec, n_fft):
    n_fft = int(n_fft)
    
    # KBD
    if isinstance(win_spec, tuple) and win_spec[0] == 'kbd':
        try:
            beta = win_spec[1]
            return kbd_window(n_fft, beta)
        except Exception as e:
            st.warning(f"KBD window failed (n_fft={n_fft}), using Kaiser window instead. Error: {e}")
            return get_window_librosa(('kaiser', win_spec[1]), n_fft).astype(np.float32)
    
    # Window
    if callable(win_spec):
        # Custom
        return win_spec(n_fft)
    else:
        # Librosa
        return get_window_librosa(win_spec, n_fft).astype(np.float32)

# ------------------------- utilities & filters ---------------------------
def stft(x, n_fft, hop, win):
    return librosa.stft(x, n_fft=n_fft, hop_length=hop, window=win, dtype=np.complex64)

def istft(Z, hop, win, length):
    return librosa.istft(Z, hop_length=hop, window=win, length=length, dtype=np.float32)

def butter_highpass(cut, sr, order=6):
    sos = butter(order, cut, btype='highpass', fs=sr, output='sos')
    return sos

def highpass(x, sr, cut=80, order=6):
    if cut <= 0: return x
    sos = butter_highpass(cut, sr, order)
    result = sosfiltfilt(sos, x, axis=0)
    del sos
    return result
    
def lowpass_env(env, sr_env, cutoff=30, order=2):
    if cutoff <= 0: return env
    if cutoff >= sr_env / 2: cutoff = 0.99 * (sr_env / 2)
    sos = butter(order, cutoff, btype='lowpass', fs=sr_env, output='sos')
    result = sosfiltfilt(sos, env, axis=1)
    del sos
    return result

# ------------------- core -------------------
def vocode(mod, car, sr, *, bands=100, n_fft=2048, hop=None, win='hann', env_lpf_cut=30):
    if hop is None: hop = n_fft // 8
    
    # Modulator envelope extraction
    M = stft(mod.astype(np.float32), n_fft, hop, win)
    
    # Carrier: Stereo STFT
    if car.ndim == 1:
        C = [stft(car.astype(np.float32), n_fft, hop, win)]
        num_channels = 1
    else:
        num_channels = car.shape[1]
        C = []
        for ch in range(num_channels):
            C.append(stft(car[:, ch].astype(np.float32), n_fft, hop, win))

    F, T = M.shape
    if bands > F: bands = F

    edges = np.linspace(0, F, bands + 1, dtype=int)
    M_mag = np.abs(M)
    env = np.zeros((bands, T), dtype=np.float32)

    for b in range(bands):
        sl = slice(edges[b], edges[b + 1])
        env[b] = M_mag[sl].mean(axis=0)

    del M, M_mag
    gc.collect()

    env = lowpass_env(env, sr / hop, cutoff=env_lpf_cut)

    # Apply envelope to each channel
    out_audio = []
    for ch_idx, S in enumerate(C):
        S_mag = np.abs(S).astype(np.float32)
        S_ph = np.angle(S)
        
        for b in range(bands):
            sl = slice(edges[b], edges[b + 1])
            car_avg = S_mag[sl].mean(axis=0) + 1e-12
            gain = env[b] / car_avg
            S_mag[sl] *= gain
        
        # Create complex array and perform ISTFT
        S_reconstructed = S_mag * np.exp(1j * S_ph)
        y_ch = istft(S_reconstructed, hop, win, len(mod))
        out_audio.append(y_ch)
        
        del S, S_mag, S_ph, S_reconstructed, y_ch
        gc.collect()

    del env, C
    gc.collect()
    
    # Output
    if num_channels == 1:
        return out_audio[0]
    else:
        return np.stack(out_audio, axis=-1)

# --------------------------- Audio Loading Function ------------------------------
def load_audio_file(file_path_or_buffer, max_duration=10):
    try:
        if isinstance(file_path_or_buffer, str):
            # File path
            audio, sr = librosa.load(file_path_or_buffer, sr=None, mono=False, dtype=np.float32)
        else:
            # File buffer
            # Reset pointer
            file_path_or_buffer.seek(0)
            audio, sr = sf.read(file_path_or_buffer, always_2d=False, dtype='float32')
        
        # Limit length
        max_samples = int(sr * max_duration)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def audio_to_bytes(audio, sr):
    try:
        audio = np.clip(audio, -1.0, 1.0)
        
        if audio.ndim == 1:
            audio_data = audio
        elif audio.ndim == 2:
            if audio.shape[0] < audio.shape[1]:
                audio_data = audio.T
            else:
                audio_data = audio
        else:
            st.error(f"Unexpected audio shape: {audio.shape}")
            return None
        
        buffer = io.BytesIO()
        from scipy.io import wavfile
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(buffer, sr, audio_int16)
        
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        st.error(f"Error converting audio to bytes: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# --------------------------- Processing Function ------------------------------
def process_vocoder(mod_file, car_file, bands, n_fft, hop_percent, 
                   win_name, kaiser_beta, tukey_alpha,
                   env_lpf_cut, modulator_locut, carrier_locut, output_locut):
    mod = car = y = None
    try:
        if mod_file is None or car_file is None:
            st.error("Please upload both Modulator and Carrier files.\nモジュレーターとキャリアーの両方のファイルをアップロードしてください")
            return None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading audio files...")
        progress_bar.progress(0)
        
        if isinstance(mod_file, str):
            mod, sr_m = librosa.load(mod_file, sr=None, mono=True, dtype=np.float32)
        else:
            mod_file.seek(0)
            mod, sr_m = librosa.load(mod_file, sr=None, mono=True, dtype=np.float32)
        
        if isinstance(car_file, str):
            car, sr_c = sf.read(car_file, always_2d=True, dtype='float32')
        else:
            car_file.seek(0)
            car, sr_c = sf.read(car_file, always_2d=True, dtype='float32')
        
        MAX_DURATION = 20
        max_samples_m = int(sr_m * MAX_DURATION)
        max_samples_c = int(sr_c * MAX_DURATION)
        
        if len(mod) > max_samples_m:
            st.info(f"Modulator truncated from {len(mod)/sr_m:.1f}s to {MAX_DURATION}s")
            mod = mod[:max_samples_m]
        if len(car) > max_samples_c:
            st.info(f"Carrier truncated from {len(car)/sr_c:.1f}s to {MAX_DURATION}s")
            car = car[:max_samples_c]
        
        # Convert carrier into 2ch stereo
        if car.shape[1] == 1:
            car = np.tile(car, (1, 2))
        
        status_text.text("Resampling...")
        progress_bar.progress(10)
        
        if sr_m != sr_c: 
            car_resampled = librosa.resample(y=car.T, orig_sr=sr_c, target_sr=sr_m, res_type='kaiser_fast').T
            del car
            car = car_resampled
            gc.collect()
        sr = sr_m
        
        status_text.text("Matching length")
        progress_bar.progress(20)
        
        N = min(len(mod), len(car))
        mod, car = mod[:N], car[:N]
        
        status_text.text("Applying pre-filters")
        progress_bar.progress(30)
        
        mod = highpass(mod, sr, cut=modulator_locut)
        car = highpass(car, sr, cut=carrier_locut)
        
        hop = int(n_fft * hop_percent / 100)
        if hop <= 0:
            st.error("hop(%) must be greater than 0.\nhop(%)は0より大きい値を指定してください")
            return None
        
        win_spec = generate_window_spec(win_name, kaiser_beta, tukey_alpha)
        win_array = generate_window_array(win_spec, n_fft)
        
        status_text.text("Vocoding...")
        progress_bar.progress(50)
        
        y = vocode(mod, car, sr, bands=int(bands), n_fft=int(n_fft), hop=hop, win=win_array, env_lpf_cut=env_lpf_cut)
        
        del mod, car, win_array
        gc.collect()
        
        status_text.text("Applying post-filter")
        progress_bar.progress(90)
        
        y = highpass(y, sr, cut=output_locut)
        y = np.clip(y, -1.0, 1.0)
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        result = (sr, y.copy())
        del y
        gc.collect()
        
        return result
    
    except Exception as e:
        del mod, car, y
        gc.collect()
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# --------------------------- Plotting Function ------------------------------
def plot_window_shape(win_name, n_fft, kaiser_beta, tukey_alpha):
    if plt is None: return None
    
    fig = None
    try:
        plt.close('all')
        
        win_spec = generate_window_spec(win_name, kaiser_beta, tukey_alpha)
        window_array = generate_window_array(win_spec, n_fft)
        
        fig, ax = plt.subplots(figsize=(8, 3), dpi=72)
        ax.plot(window_array, linewidth=1)
        ax.set_title(f"'{win_name.capitalize()}' Window (N={int(n_fft)})", fontsize=10)
        ax.set_xlabel("Sample", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        
        del window_array
        return fig
        
    except Exception as e:
        st.warning(f"Plot error: {e}")
        if fig is not None:
            plt.close(fig)
        return None
    finally:
        gc.collect()

# --------------------------- Main UI ------------------------------
def main():
    st.title("🎵 Vococo FFT Vocoder")
    
    WINDOW_FUNCTIONS = {
        "Kaiser": "kaiser", "Hann": "hann", "Hamming": "hamming", 
        "Blackman": "blackman", "Vorbis": "vorbis", "Welch": "welch", 
        "Tukey": "tukey", "Bartlett": "bartlett", "Blackman-Harris": "blackmanharris", 
        "Kaiser-Bessel Derived": "kbd", "Flat Top": "flattop", 
        "Sine": "cosine", "Lanczos": "lanczos", "Rectangular": "boxcar", 
        "Bohman": "bohman", "Nuttall": "nuttall", "Parzen": "parzen"
    }
    
    MOD_DEFAULT_PATH = "modulator.wav"
    CAR_DEFAULT_PATH = "carrier.wav"
    default_hop_percent = (75 / 768) * 100
    
    # Loading default file
    if 'default_files_loaded' not in st.session_state:
        st.session_state.default_files_loaded = False
    
    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Audio Input")
        
        # Modulator
        mod_uploaded = st.file_uploader("Modulator", type=["wav", "mp3", "flac", "ogg"], key="mod_uploader")
        
        # If default modulator available
        if mod_uploaded is None and Path(MOD_DEFAULT_PATH).exists():
            #st.info(f"Using default: {MOD_DEFAULT_PATH}")
            mod_file = MOD_DEFAULT_PATH
            # Preview
            st.markdown("**Preview Modulator (Default)**")
            mod_audio, mod_sr = load_audio_file(MOD_DEFAULT_PATH)
            if mod_audio is not None:
                audio_bytes = audio_to_bytes(mod_audio, mod_sr)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/wav')
                    st.caption(f"Duration: {len(mod_audio)/mod_sr:.2f}s | Sample Rate: {mod_sr} Hz")
        elif mod_uploaded is not None:
            mod_file = mod_uploaded
            # Uploaded file preview
            st.markdown("**Preview Modulator (Uploaded)**")
            mod_audio, mod_sr = load_audio_file(mod_uploaded)
            if mod_audio is not None:
                audio_bytes = audio_to_bytes(mod_audio, mod_sr)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/wav')
                    st.caption(f"Duration: {len(mod_audio)/mod_sr:.2f}s | Sample Rate: {mod_sr} Hz")
        else:
            mod_file = None
        
        # Carrier
        car_uploaded = st.file_uploader("Carrier", type=["wav", "mp3", "flac", "ogg"], key="car_uploader")
        
        # If default carrier available
        if car_uploaded is None and Path(CAR_DEFAULT_PATH).exists():
            #st.info(f"Using default: {CAR_DEFAULT_PATH}")
            car_file = CAR_DEFAULT_PATH
            # Preview
            st.markdown("**Preview Carrier (Default)**")
            car_audio, car_sr = load_audio_file(CAR_DEFAULT_PATH)
            if car_audio is not None:
                audio_bytes = audio_to_bytes(car_audio, car_sr)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/wav')
                    st.caption(f"Duration: {len(car_audio)/car_sr:.2f}s | Sample Rate: {car_sr} Hz")
        elif car_uploaded is not None:
            car_file = car_uploaded
            # Uploaded file preview
            st.markdown("**Preview Carrier (Uploaded)**")
            car_audio, car_sr = load_audio_file(car_uploaded)
            if car_audio is not None:
                audio_bytes = audio_to_bytes(car_audio, car_sr)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/wav')
                    st.caption(f"Duration: {len(car_audio)/car_sr:.2f}s | Sample Rate: {car_sr} Hz")
        else:
            car_file = None
        
        with st.expander("Core Parameters", expanded=True):
            bands = st.slider("bands", 1, 500, 100, step=1, 
                            help="Number of frequency bands for the vocoder.\nボコーダーの周波数帯域の数")
            n_fft = st.slider("n_fft", 64, 8192, 768, step=64, 
                            help="FFT size (max 8192 for memory).\nFFTサイズ (メモリ制限: 最大8192)")
            hop_percent = st.slider("hop (%)", 0.1, 100.0, default_hop_percent, 
                                  help="Overlap of FFT frames.\nFFTフレームの移動量")
    
    with col2:
        with st.expander("Window Parameters", expanded=True):
            win_name = st.selectbox("Window Function", list(WINDOW_FUNCTIONS.values()), 
                                   index=list(WINDOW_FUNCTIONS.values()).index('vorbis'))
            
            is_kaiser_type = win_name in ["kaiser", "kbd"]
            is_tukey = win_name == "tukey"
            
            kaiser_beta = 14.0
            tukey_alpha = 0.5
            
            if is_kaiser_type:
                kaiser_beta = st.slider("Kaiser / KBD Beta", 0.1, 20.0, 14.0, step=0.1,
                                       help="Adjusts the trade-off between main-lobe width and side-lobe level.\nサイドローブの減衰量を調整します")
            if is_tukey:
                tukey_alpha = st.slider("Tukey Alpha", 0.0, 1.0, 0.5, step=0.01,
                                       help="Ratio of the tapered section (0=Rectangular, 1=Hann).\n窓の平坦な部分の割合 (0=矩形, 1=Hann)")
        
        with st.expander("Window Shape Preview", expanded=True):
            window_plot_placeholder = st.empty()
            fig = plot_window_shape(win_name, n_fft, kaiser_beta, tukey_alpha)
            if fig:
                window_plot_placeholder.pyplot(fig)
                plt.close(fig)
                
        with st.expander("Filter Parameters", expanded=False):
            modulator_locut = st.slider("modulator locut (Hz)", 0, 500, 160, step=1,
                                       help="High-pass filter for the Modulator input.\nModulator入力のハイパスフィルター")
            carrier_locut = st.slider("carrier locut (Hz)", 0, 500, 160, step=1,
                                     help="High-pass filter for the Carrier input.\nCarrier入力のハイパスフィルター")
            output_locut = st.slider("output locut (Hz)", 0, 500, 128, step=1,
                                    help="High-pass filter for the final output.\n最終出力のハイパスフィルター")
            env_lpf_cut = st.slider("env_lpf_cut (Hz)", 0, 500, 300, step=1,
                                   help="Low-pass filter for the envelope.\n包絡線に適用するローパスフィルター")
    
    # Vocode Button
    st.markdown("---")
    if st.button("Vocode", type="primary", use_container_width=True):
        result = process_vocoder(
            mod_file, car_file, bands, n_fft, hop_percent, 
            win_name, kaiser_beta, tukey_alpha,
            env_lpf_cut, modulator_locut, carrier_locut, output_locut
        )
        
        if result:
            sr, audio = result
            st.success("Processing complete!")
            st.subheader("Output Audio")
            
            # Convert audio to bytes
            audio_bytes = audio_to_bytes(audio, sr)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/wav')
                
                # Verbose
                duration = len(audio) / sr if audio.ndim == 1 else len(audio) / sr
                channels = "Mono" if audio.ndim == 1 else f"Stereo ({audio.shape[1]} channels)"
                st.caption(f"Duration: {duration:.2f}s | Sample Rate: {sr} Hz | {channels}")
    
    #st.warning("⚠️ Max 20 seconds of audio | 最大20秒に制限中")

if __name__ == "__main__":
    main()