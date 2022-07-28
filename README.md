# deep-stroke
Implementation of a recognizer for stroke gestures based on Long Short Term Memory Recurrent Neural Networks.

## Dataset reference
The datasets used for the experiments can be found at https://luis.leiva.name/g3/ in the dataset section. Each dataset has to be inserted in the corresponding folder as shown in the `DataLoader` file. For both the singlestroke and the multistroke we used the synthetic best reconstruction for training and the human samples for test.

- dataset
  - GGG
    - 1$
      - csv-1dollar-human
      - csv-1dollar-synth-best
    - N$
      - csv-ndollar-human
      - csv-ndollar-synth-best
