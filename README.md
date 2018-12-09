# Secret sharer and other words

## Tech specs
  - **Environment**: Python 3.7.0; TensorFlow 1.12.0, run on a Macbook Pro CPU; Keras 2.2.4; Pandas 0.23.4; NumPy 1.15.4;
  - **Summary of typical model**:  ```_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_14 (Embedding)     (None, 4, 4)              56336     
_________________________________________________________________
lstm_27 (LSTM)               (None, 4, 100)            42000     
_________________________________________________________________
lstm_28 (LSTM)               (None, 100)               80400     
_________________________________________________________________
dense_27 (Dense)             (None, 100)               10100     
_________________________________________________________________
dense_28 (Dense)             (None, 14084)             1422484   
=================================================================
Total params: 1,611,320
Trainable params: 1,611,320
Non-trainable params: 0
_________________________________________________________________```

## Differences from Carlini paper

## Code walkthrough
