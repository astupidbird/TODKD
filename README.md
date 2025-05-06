#  TODKD

## Requiements
### Environments
- `python 3.8`
- `PyTorch 1.7.1`
- `transformers 4.10.0`
- `numpy 1.24.4`
- `scikit-learn 0.24.2`
- `openai 0.28.0`
- `nltk 3.8.1`
- `retry 0.9.2`

Please install all the dependency packages using the following command:
```
conda create -n TODKD python=3.8
conda activate TODKD
pip install -r requirements.txt
```

### Encoding models
By default, BERT is our encoder model, please download pretrained BERT from [huggingface](https://huggingface.co/models) and put it into the `./bert-base-uncased` directory.

You can also choose RoBERTa as the encoder model, please download pretrained RoBERTa from [huggingface](https://huggingface.co/models) and put it into the `./roberta-base` directory.

### Qwen
You need to first apply for an [QwenAI account](https://www.alibabacloud.com/help/zh/model-studio/what-is-qwen-llm), and then buy the Qwen API to get your own **API key**. Then set your key in `config.ini`.




## Datasets
We conduct our experiments on two public relation extraction datasets:
- [FewRel](https://github.com/thunlp/FewRel)
- [TACRED](https://nlp.stanford.edu/projects/tacred/)


## Train
To run our method, use command: 
 ```
  bash bash/fewrel_5shot.sh  # for FewRel 5-shot setting
  bash bash/fewrel_10shot.sh # for FewRel 10-shot setting
  bash bash/tacred_5shot.sh  # for TACRED 5-shot setting
  bash bash/tacred_10shot.sh # for TACRED 10-shot setting
```

You can refer to `config.ini` to adjust other hyperparameters.
