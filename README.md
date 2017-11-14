# Code for <i> Frame-Based Continuous Lexical Semantics through Exponential Family Factorization and Semantic Proto-Roles </i>

### Paper
The paper can be found [here.](http://aclweb.org/anthology/S/S17/S17-1011.pdf)

### Instructions
<i>TODO:</i> Add instructions on how to generate embeddings based on our <i>n</i>-way tensor factorization.

### Important files:
1. ```word2vecT.c``` - our modifications to [Cotterel et. al. (2017)'s](https://aclweb.org/anthology/E/E17/E17-2028.pdf) [word2vec3.c](https://github.com/azpoliak/skip-gram-tensor/blob/master/hyperref-tensor/word2vec3.c) tensor factorization code to generate word embeddings


### Citation
If make use of our code, please use the following citation:

```
@InProceedings{ferraro-EtAl:2017:starSEM,
  author    = {Ferraro, Francis  and  Poliak, Adam  and  Cotterell, Ryan  and  Van Durme, Benjamin},
  title     = {Frame-Based Continuous Lexical Semantics through Exponential Family Tensor Factorization and Semantic Proto-Roles},
  booktitle = {Proceedings of the 6th Joint Conference on Lexical and Computational Semantics (*SEM 2017)},
  month     = {August},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {97--103},
  abstract  = {We study how different frame annotations complement one another when learning
	continuous lexical semantics. We learn the representations from a tensorized
	skip-gram model that consistently encodes syntactic-semantic content better,
	with multiple 10% gains over baselines.},
  url       = {http://www.aclweb.org/anthology/S17-1011}
}
``` 
