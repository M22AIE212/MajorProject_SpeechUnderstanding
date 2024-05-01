import os
import errno
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import librosa
import librosa.display


def fn_directory_create(speaker_list,dir_name = "train-gram") :

    # Create directories for speakers
    for speaker in speaker_list:
      try:
          os.makedirs(root + f'/{dir_name}/' + str(speaker))
          print('Created. Directory for Speaker {}'.format(speaker))
      except OSError as e:
          if e.errno != errno.EEXIST:
              raise

    print('All Directories Created .. ')

def fn_data_prep(data, dpi,dir_name = "train-gram"):
    """
    Prepare data for processing.

    Args:
    - data: DataFrame containing data for processing
    - dpi: Dots per inch for saving the spectrogram images

    Result:
    -  Slice audio into segments and generate spectrograms
    """

    for index, row in data.iterrows():
        dir_path = root + '/' + row['SUBSET'] + '/' + str(row['ID']) + '/'
        print('Working on DataFrame row {}, speaker {}'.format(index, row['CODE']))
        if not os.path.exists(dir_path):
            print('Directory {} does not exist, skipping'.format(dir_path))
            continue

        # Get all FLAC files in the directory
        files_iter = Path(dir_path).glob('**/*.flac')
        files_list = [str(f) for f in files_iter]

        # Process each audio file
        for file_path in files_list:
            audio, sample_rate = librosa.load(file_path)
            duration = audio.shape[0] / sample_rate
            start = 0

            # Slice audio into segments and generate spectrograms
            while start + 5 < duration:
                segment = audio[start * sample_rate: (start + 5) * sample_rate]
                start += 5 - 1
                spectrogram = librosa.stft(segment)
                spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
                plt.figure(figsize=(227 / dpi, 227 / dpi), dpi=dpi)
                plt.axis('off')
                librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='log')
                plt.savefig("/content/" + root + f'/{dir_name}/' + str(row['CODE']) + '/' + uuid.uuid4().hex + '.png', dpi=dpi)
                plt.close()

    print('Data preparation completed.')

if __name__ == "__main__" :

  root = 'LibriSpeech'
  dir_name = "train-gram"
  dpi = 120
  num_speakers = 10

  ## Loading Meta Data
  speakers = pd.read_csv(root + '/SPEAKERS.TXT', sep='|', on_bad_lines='skip')

  ## Cleaning Meta Data File
  speakers.columns = [col.strip() for col in speakers.columns ]
  speakers = speakers.applymap(lambda x: x.strip() if isinstance(x, str) else x)

  speakers_subset = speakers[(speakers['SUBSET'] == 'test-clean')]
  speakers_subset['CODE'] = speakers_subset['NAME'].astype('category').cat.codes
  speaker_list = np.unique(speakers_subset['CODE'])[:num_speakers]
  speakers_subset = speakers_subset[speakers_subset.CODE.isin(speaker_list)].sort_values("CODE").reset_index(drop = True)

  print("Creating Directories for all speakers : ")
  fn_directory_create(speaker_list,dir_name)

  print("Spectrogram Creation ... ")
  fn_data_prep(speakers_subset, dpi,dir_name)
