# CUD-NET: Color Universal Design Neural Filter for the Color Weakness

This repository is the official implementation of [CUD-NET: Color Universal Design Neural Filter for
the Color Weakness](https://arxiv.org/abs/2030.12345).

Sunyong Seo, Hana Kim, Jinho Park

# Notice
- As our train dataset is exclusive to cooperation institutes, only part of the inference dataset can be released.
- The HSV conversion algorithm has defunction of resversing Hue on the value with 180.
- We used extra training data on up-loaded model, so there would be some error comparing to papers'

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training and Evaluation

```
python main.py --log_interval 100 --save_interval 10 -cuda -pin_memory --mode train --worker 8
python test_image.py -cuda -pin_memory --worker 2 --mode inference --data_class A
```


## Pre-trained Models

You can find pretrained models on "<i>model_checkpoints/</i>":


## Results

![image_1](images/image_01.jpg)

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)