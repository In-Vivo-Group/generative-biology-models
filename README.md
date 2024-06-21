# generative-biology-models

Collection of generative biology models

## Ankh
Ankh is the first general-purpose protein language model trained on Google's TPU-V4 surpassing the state-of-the-art performance with dramatically less parameters, promoting accessibility to research innovation via attainable resources.

[Paper](https://arxiv.org/abs/2301.06568)
[Repo](https://github.com/agemagician/Ankh)

## ESM series
### ESM-1b
A transformer protein language model, trained on protein sequence data. The model is trained to predict amino acids from the surrounding sequence context
[Paper](https://www.pnas.org/content/118/15/e2016239118)
[Repo](https://github.com/facebookresearch/esm)

### ESM-2 
ESM-2 is a state-of-the-art protein model trained on a masked language modelling objective. ESM-2 outperforms all tested single-sequence protein language models across a range of structure prediction tasks, and enables atomic resolution structure prediction.
[Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
[Repo](https://github.com/facebookresearch/esm)

### ESMFold
ESMFold is a state-of-the-art end-to-end protein folding model based on an ESM-2 backbone
[Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
[Repo](https://github.com/facebookresearch/esm)

### OpenFold
Trainable, memory-efficient, and GPU-friendly PyTorch reproduction of AlphaFold 2
[Paper](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v1)
[Repo](https://github.com/aqlaboratory/openfold)

### ProstT5
ProstT5 is a protein language model (pLM) which can translate between protein sequence and structure.
[Paper](https://www.biorxiv.org/content/10.1101/2023.07.23.550085v1)
[Repo](https://huggingface.co/Rostlab/ProstT5)

### ProtT5-XL
Pretrained model on protein sequences using a masked language modeling (MLM) objective.
[Paper](https://doi.org/10.1101/2020.07.12.199554)
[Repo](https://github.com/agemagician/ProtTrans)

### ProtBert
Pretrained model on protein sequences using a masked language modeling (MLM) objective. This model is trained on uppercase amino acids: it only works with capital letter amino acids.
[Paper](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3)
[Repo](https://github.com/agemagician/ProtTrans)

### ProtGPT2
ProtGPT2 (peer-reviewed paper) is a language model that speaks the protein language and can be used for de novo protein design and engineering. ProtGPT2 generated sequences conserve natural proteins' critical features (amino acid propensities, secondary structural content, and globularity) while exploring unseen regions of the protein space.
[Paper](https://www.nature.com/articles/s41467-022-32007-7)
[Repo](https://huggingface.co/nferruz/ProtGPT2)




