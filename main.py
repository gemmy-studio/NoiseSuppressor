import os
import glob
import torchaudio
import torchaudio.transforms as T
import numpy as np
from matplotlib import pyplot as plt
import librosa
import librosa.display
from df import enhance, init_df
import streamlit as st
# from streamlit.components.v1 import html

# @inproceedings{schroeter2023deepfilternet3,
#   title = {{DeepFilterNet}: Perceptually Motivated Real-Time Speech Enhancement},
#   author = {Schröter, Hendrik and Rosenkranz, Tobias and Escalante-B., Alberto N. and Maier, Andreas},
#   booktitle={INTERSPEECH},
#   year = {2023},
# }
# -----------------------------------------------------------------------------------------

app_title = "소음 억제 도구"
model, df_state, _ = init_df()  # Load default model
df_sr = 48000


def display_audio_info(audio, title):
    # 두 개의 컬럼 생성
    col1, col2 = st.columns(2)

    audio = np.clip(audio, -1.0, 1.0)
    if len(np.shape(audio)) == 2:
        audio = audio[0]

    # 왼쪽 컬럼에 스펙트로그램 표시
    with col1:
        st.markdown(f"### {title} - Spectrogram")
        D = librosa.stft(audio)  # STFT of y
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(
            S_db, x_axis='time', y_axis='linear', ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        st.pyplot(fig)

    # 오른쪽 컬럼에 파형 표시
    with col2:
        st.markdown(f"### {title} - Waveform")
        fig, ax = plt.subplots()
        plt.plot(audio)
        ax.set_xticks([])
        ax.set_ylim(-1, 1)
        st.pyplot(fig)


def main():
    st.set_page_config(page_title=app_title, page_icon="favicon.ico",
                       layout="centered", initial_sidebar_state="auto", menu_items=None)

    st.title(app_title)
    st.divider()
    st.header('손쉽게 불필요한 소음을 제거하세요!')

    uploaded_file = st.file_uploader(
        "변환할 파일을 업로드 해주세요. (지원 형식: .wav, .mp3, .opus)")

    if uploaded_file:
        # 이전에 다운로드 한 파일을 삭제
        files_to_remove = glob.glob('enhanced_*')
        for file in files_to_remove:
            os.remove(file)

        uploaded_file_type = uploaded_file.type.split('/')[-1]
        print(uploaded_file_type)
        if uploaded_file_type not in ['wav', 'mpeg', 'ogg']:
            st.text('지원하지 않는 파일 형식입니다.')
        else:
            with st.spinner('소음 제거하는 중'):
                noisy_audio, sr = torchaudio.load(uploaded_file)
                print("np.shape(noisy_audio)", np.shape(noisy_audio))
                st.audio(noisy_audio.numpy(), sample_rate=sr)

                # 샘플링 레이트가 48000Hz가 아닐 경우 리샘플링
                if sr != df_sr:
                    resampler = T.Resample(orig_freq=sr, new_freq=df_sr)
                    noisy_audio = resampler(noisy_audio)
                display_audio_info(noisy_audio.numpy(), "입력")

            with st.spinner('소음 제거하는 중'):
                output_audio = enhance(model, df_state, noisy_audio)
                enhanced_audio = output_audio
                st.divider()
                # 샘플링 레이트가 48000Hz가 아닐 경우 리샘플링
                if sr != df_sr:
                    resampler = T.Resample(orig_freq=df_sr, new_freq=sr)
                    enhanced_audio = resampler(enhanced_audio)
                st.audio(enhanced_audio.numpy(), sample_rate=sr)
                display_audio_info(output_audio.numpy(), "출력")


if __name__ == '__main__':
    main()
