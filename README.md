<h1 align="center">OpenAI Whisper를 활용한 화자 대화</h1>

# OpenAI Whisper 기반 화자 분류 파이프라인

## 목적
이 저장소는 Whisper ASR 기능과 음성 활동 감지(VAD) 및 화자 분류를 결합하여 Whisper에서 생성된 결과물에서 각 문장의 화자를 식별합니다. 진행 과정은 다음과 같습니다.

1. 화자 분류 정확도를 높이기 위해 오디오에서 음성을 추출
2. Whisper를 사용하여 전사본을 생성
3. 타임스팸프를 수정하고 정렬
4. VAD 및 무음 제거를 위한 분할을 위해 MarbelNet으로 전달
5. TitaNet을 사용하여 각 세그먼트의 화자를 식별하기 위한 화자 임베딩 추출
6. 생성된 타임스탬프와 연관하여 타임스팸트 기반으로 각 단어의 화자를 감지
7. 구두점 모델을 사용하여 미세한 시간 이동 보정을 위해 재정렬

## 설치
Python >= `3.10` 가 필요하고, `3.9` 에서도 작동하지만 requirements를 하나씩 수동 설치해야 합니다. (3.10 이상 권고)
`FFMPEG`과 `Cython`을 사전에 설치해야 합니다.

```
pip install cython
```
또는
```
sudo apt update && sudo apt install cython3
```
```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg

# on Windows using WinGet (https://github.com/microsoft/winget-cli)
winget install ffmpeg
```
```
pip install -c constraints.txt -r requirements.txt
```
## 사용법

```
python diarize.py -a AUDIO_FILE_NAME
```

시스템에 충분한 VRAM(10GB 이상)이 있다면 diarize_parallel.py 대신 사용할 수 있습니다. 차이점은 NeMo를 Whisper와 병렬로 실핸한다는 것입니다.
이는 경우에 따라 유용할 수 있으며, 두 모델이 서로 종속적이지 않기 때문에 결과는 동일합니다. 아직 실험 단계이므로 오류와 문제가 있을 수 있습니다.

## 명령 옵션

- `-a AUDIO_FILE_NAME`: 처리할 오디오 파일 이름
- `--no-stem`: 소스 분리를 비활성화합니다.
- `--whisper-model`: ASR에 사용할 모델, 기본값은 다음과 같습니다. `medium.en`
- `--suppress_numerals`: 숫자 대신 발음된 문자를 숫자로 표기하여 정렬 정확도를 향상시킵니다.
- `--device`: 사용할 장치를 선택합니다. 사용 가능한 경우 기본값은 "cuda"입니다.
- `--language`: 언어 감지에 실패한 경우 언어를 수동으로 선택합니다.
- `--batch-size`: 추론의 배치 크기, 메모리가 부족하면 줄이고, 배치되지 않은 추론의 경우 0으로 설정합니다.

## 알려진 제약 사항
- 오류가 있을 수 있으니, 오류가 발견되면 문제를 게지해 주시기 바랍니다.

## 향후 개선 사항
- SRT에 대한 문장당 최대 길이 구현

## 감사의 말
Special Thanks for [@adamjonas](https://github.com/adamjonas) for supporting this project
This work is based on [OpenAI's Whisper](https://github.com/openai/whisper) , [Faster Whisper](https://github.com/guillaumekln/faster-whisper) , [Nvidia NeMo](https://github.com/NVIDIA/NeMo) , and [Facebook's Demucs](https://github.com/facebookresearch/demucs)

## Citation
연구에 이것을 사용할 경우 프로젝트를 인용해 주세요.

```bibtex
@unpublished{hassouna2024whisperdiarization,
  title={Whisper Diarization: Speaker Diarization Using OpenAI Whisper},
  author={Ashraf, Mahmoud},
  year={2024}
}
```
