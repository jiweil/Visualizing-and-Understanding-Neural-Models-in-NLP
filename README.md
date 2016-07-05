# Visualizing and Understanding Neural Models in NLP
Implementations of two saliency models described in "Visualizing and Understanding Neural Models in NLP" by Jiwei Li, Xinlei Chen, Eduard Hovy and Dan Jurafsky.

## Requirements:
GPU
Torch (nn,cutorch,cunn,nngraph)
[python matplotlib library](http://matplotlib.org/) (only for matrix plotting purposes)

## Run the models:
Visualization using the first-derivative saliency model:

sh saliency_derivative.sh

Visualization using the variance saliency model:

sh saliency_variance.sh

After running, the saliency matrix will be stored in the file "matrix"

##Folders and Files
input.txt: the input sentence.

sentiment_bidi: training bi-directional lstms on the stanford sentiment treebank. Readers can either directly load a pretrained model (sentiment_bidi/model) or train it by yourself by running sentiment_bidi/main.lua

data/dict.txt: word dictionary. Current models only support tokens found in the dictionary. Will fix it soon.

download [data](http://cs.stanford.edu/~bdlijiwei/visual_data.tar)

For any pertinent questions, feel free to contact jiweil@stanford.edu

## Acknowledgments

[Yoon Kim's seq2seq-attn repo](https://github.com/harvardnlp/seq2seq-attn)

[Wojciech Zaremba's lstm repo](https://github.com/wojzaremba/lstm)

[Socher et al.,'s Stanford Sentiment Treebank dataset](http://nlp.stanford.edu/sentiment/index.html)

```latex
@article{li2015visualizing,
    title={Visualizing and understanding neural models in NLP},
    author={Li, Jiwei and Chen, Xinlei and Hovy, Eduard and Jurafsky, Dan},
    journal={arXiv preprint arXiv:1506.01066},
    year={2015}
}
