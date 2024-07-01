# generative-biology-models

Collection of generative biology models

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/samwell/genBio)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S1H8a2i-ETLCT43mFwX2Q4yboHrtcc9X)

Refer to our [paper](https://in-vivo-group.github.io/generative-biology/) for a detailed description of these models as seen in [Protein large language models](https://in-vivo-group.github.io/generative-biology/#protein-large-language-models-prot-llms.) and [Genomic large language models](https://in-vivo-group.github.io/generative-biology/#genomic-large-language-models-gene-llms). Our focus is on prompting and generating novel sequences so we'll be mostly considering decoder-based and encoder-decoder based models.

## Protein large language models
- [Decoder-based models](#decoder-based-models)
  - ProGen
  - ProGen2
  - ProtGPT2
  - RITA
  - PoET
  - LM-Design
  - ZymCTRL
  - IgLM

- [Encoder-decoder based models](#encoder-decoder-based-models)
  - ProstT5
  - pAbT5
  - xTrimoPGLM
  - Small-Scale protein Language Model (SS-pLM)
  - MSA2Prot
  - MSA-Augmenter
  - Fold2Seq

## Decoder-based models

Details about decoder-based models go here.


Details about decoder-based models go here.




## ProtGPT2
ProtGPT2 is a language model that speaks the protein language and can be used for de novo protein design and engineering. ProtGPT2 generated sequences conserve natural proteins' critical features (amino acid propensities, secondary structural content, and globularity) while exploring unseen regions of the protein space.
[Paper](https://www.nature.com/articles/s41467-022-32007-7)
[Repo](https://huggingface.co/nferruz/ProtGPT2)

<details>
  <summary>Sample code</summary>

  ### Generating de novo proteins in a zero-shot fashion

  In the example below, ProtGPT2 generates sequences that follow the amino acid 'M'. Any other amino acid, oligomer, fragment, or protein of choice can be selected instead. The model      will generate the most probable sequences that follow the input. Alternatively, the input field can also be left empty and it will choose the starting tokens.
  
  ```py
  >>> from transformers import pipeline
  >>> protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")
  >>> sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)
  >>> for seq in sequences:
          print(seq):

  {'generated_text': 'MINDLLDISRIISGKMTLDRAEVNLTAIARQVVEEQRQAAEAKSIQLLCSTPDTNHYVFG\nDFDRLKQTLWNLLSNAVKFTPSGGTVELELGYNAEGMEVYVKDSGIGIDPAFLPYVFDRF\nRQSDAADSRNYGGLGLGLAIVKHLLDLHEGNVSAQSEGFGKGATFTVLLPLKPLKRELAA\nVNRHTAVQQSAPLNDNLAGMKILIVEDRPDTNEMVSYILEEAGAIVETAESGAAALTSLK\nSYSPDLVLSDIGMPMMDGYEMIEYIREWKTTKGG'}
{'generated_text': 'MQGDSSISSSNRMFT\nLCKPLTVANETSTLSTTRNSKSNKRVSKQRVNLAESPERNAPSPASIKTNETEEFSTIKT\nTNNEVLGYEPNYVSYDFVPMEKCNLCNENCSIELASLNEETFVKKTICCHECRKKAIENA\nENNNTKGSAVSNNSVTSSSGRKKIIVSGSQILRNLDSLTSSKSNISTLLNPNHLAKLAKN\nGNLSSLSSLQSSASSISKSSSTSSTPTTSPKVSSPTNSPSSSPINSPTP'}
{'generated_text': 'M\nSTHVSLENTLASLQATFFSLEARHTALETQLLSTRTELAATKQELVRVQAEISRADAQAQ\nDLKAQILTLKEKADQAEVEAAAATQRAEESQAALEAQTAELAQLRLEKQAPQHVAEEGDP\nQPAAPTTQAQSPVTSAAAAASSAASAEPSKPELTFPAYTKRKPPTITHAPKAPTKVALNP\nSTLSTSGSGGGAKADPTPTTPVPSSSAGLIPKALRLPPPVTPAASGAKPAPSARSKLRGP\nDAPLSPSTQS'}
{'generated_text': 'MVLLSTGPLPILFLGPSLAELNQKYQVVSDTLLRFTNTV\nTFNTLKFLGSDS\n'}
{'generated_text': 'M\nNNDEQPFIMSTSGYAGNTTSSMNSTSDFNTNNKSNTWSNRFSNFIAYFSGVGWFIGAISV\nIFFIIYVIVFLSRKTKPSGQKQYSRTERNNRDVDSIKRANYYG\n'}
{'generated_text': 'M\nEAVYSFTITETGTGTVEVTPLDRTISGADIVYPPDTACVPLTVQPVINANGTWTLGSGCT\nGHFSVDTTGHVNCLTGGFGAAGVHTVIYTVETPYSGNSFAVIDVNVTEPSGPGDGGNGNG\nDRGDGPDNGGGNNPGPDPDPSTPPPPGDCSSPLPVVCSDRDCADFDTQAQVQIYLDRYGG\nTCDLDGNHDGTPCENLPNNSGGQSSDSGNGGGNPGTGSTHQVVTGDCLWNIASRNNGQGG\nQAWPALLAANNESITNP'}
{'generated_text': 'M\nGLTTSGGARGFCSLAVLQELVPRPELLFVIDRAFHSGKHAVDMQVVDQEGLGDGVATLLY\nAHQGLYTCLLQAEARLLGREWAAVPALEPNFMESPLIALPRQLLEGLEQNILSAYGSEWS\nQDVAEPQGDTPAALLATALGLHEPQQVAQRRRQLFEAAEAALQAIRASA\n'}
{'generated_text': 'M\nGAAGYTGSLILAALKQNPDIAVYALNRNDEKLKDVCGQYSNLKGQVCDLSNESQVEALLS\nGPRKTVVNLVGPYSFYGSRVLNACIEANCHYIDLTGEVYWIPQMIKQYHHKAVQSGARIV\nPAVGFDSTPAELGSFFAYQQCREKLKKAHLKIKAYTGQSGGASGGTILTMIQHGIENGKI\nLREIRSMANPREPQSDFKHYKEKTFQDGSASFWGVPFVMKGINTPVVQRSASLLKKLYQP\nFDYKQCFSFSTLLNSLFSYIFNAI'}
{'generated_text': 'M\nKFPSLLLDSYLLVFFIFCSLGLYFSPKEFLSKSYTLLTFFGSLLFIVLVAFPYQSAISAS\nKYYYFPFPIQFFDIGLAENKSNFVTSTTILIFCFILFKRQKYISLLLLTVVLIPIISKGN\nYLFIILILNLAVYFFLFKKLYKKGFCISLFLVFSCIFIFIVSKIMYSSGIEGIYKELIFT\nGDNDGRFLIIKSFLEYWKDNLFFGLGPSSVNLFSGAVSGSFHNTYFFIFFQSGILGAFIF\nLLPFVYFFISFFKDNSSFMKLF'}
{'generated_text': 'M\nRRAVGNADLGMEAARYEPSGAYQASEGDGAHGKPHSLPFVALERWQQLGPEERTLAEAVR\nAVLASGQYLLGEAVRRFETAVAAWLGVPFALGVASGTAALTLALRAYGVGPGDEVIVPAI\nTFIATSNAITAAGARPVLVDIDPSTWNMSVASLAARLTPKTKAILAVHLWGQPVDMHPLL\nDIAAQANLAVIEDCAQALGASIAGTKVGTFGDAAAFSFYPTKNMTTGEGGMLVTNARDLA\nQAARMLRSHGQDPPTAYMHSQVGFN'}

  
  ```
</details>

<details>
<summary>How to select the best sequences</summary>

### Compute the perplexity for each sequence as follows:
Where ppl is a value with the perplexity for that sequence. Given the fast inference times, the best threshold as to what perplexity value gives a 'good' or 'bad' sequence, is to sample many sequences, order them by perplexity, and select those with the lower values (the lower the better).

```py
sequence='MGEAMGLTQPAVSRAVARLEERVGIRIFNRTARAITLTDEGRRFYEAVAPLLAGIEMHGYR\nVNVEGVAQLLELYARDILAEGRLVQLLPEWAD'

#Convert the sequence to a string like this
#(note we have to introduce new line characters every 60 amino acids,
#following the FASTA file format).

sequence = "<|endoftext|>MGEAMGLTQPAVSRAVARLEERVGIRIFNRTARAITLTDEGRRFYEAVAPLLAGIEMHGY\nRVNVEGVAQLLELYARDILAEGRLVQLLPEWAD<|endoftext|>"

# ppl function
def calculatePerplexity(sequence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0) 
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)

#And hence: 
ppl = calculatePerplexity(sequence, model, tokenizer)

```
</details>

## RITA

A suite of autoregressive generative models for protein sequences,
with up to 1.2 billion parameters, trained on over
280 million protein sequences belonging to the
UniRef-100 database. 
[Paper](https://huggingface.co/papers/2205.05789)
[Repo](https://huggingface.co/lightonai/RITA_xl)

<details>
<summary>Sample code</summary>

  ```py
  from transformers import AutoModel, AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("lightonai/RITA_xl, trust_remote_code=True")
  tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_xl")

  from transformers import pipeline
  rita_gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
  sequences = rita_gen("MAB", max_length=20, do_sample=True, top_k=950, repetition_penalty=1.2, 
                       num_return_sequences=2, eos_token_id=2)
  for seq in sequences:
      print(f"seq: {seq['generated_text'].replace(' ', '')}")

  ```
</details>

## RFDiffusion

RFdiffusion is an open source method for structure generation, with or without conditional information
[Paper](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1)
[Repo](https://github.com/RosettaCommons/RFdiffusion)

<details>

  <summary>Sample code</summary>

  ### Unconditional design of a protein
  For this, we just need to specify three things:

  1. The length of the protein
  2. The location where we want to write files to
  3. The number of designs we want

     ```py

     ./scripts/run_inference.py 'contigmap.contigs=[150-150]' inference.output_prefix=test_outputs/test inference.num_designs=10


     ```
  
</details>

## Protpardelle

An all-atom protein generative model

[Paper](https://doi.org/10.1101/2023.05.24.542194)
[Code](https://github.com/ProteinDesignLab/protpardelle)

<details>
  <summary>Sample code</summary>
  Demo: https://huggingface.co/spaces/ProteinDesignLab/protpardelle

  ### Unconditional sampling:
  For this, we just need to specify three things:

  1. The length of the protein (max and min)
  2. How frequently to select sequence length
  3. The number of samples we want
  
  
</details>



<br/>
<br/>
<br/>

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

## OpenFold
Trainable, memory-efficient, and GPU-friendly PyTorch reproduction of AlphaFold 2
[Paper](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v1)
[Repo](https://github.com/aqlaboratory/openfold)

## ProstT5
ProstT5 is a protein language model (pLM) which can translate between protein sequence and structure.
[Paper](https://www.biorxiv.org/content/10.1101/2023.07.23.550085v1)
[Repo](https://huggingface.co/Rostlab/ProstT5)

## ProtT5-XL
Pretrained model on protein sequences using a masked language modeling (MLM) objective.
[Paper](https://doi.org/10.1101/2020.07.12.199554)
[Repo](https://github.com/agemagician/ProtTrans)

## ProtBert
Pretrained model on protein sequences using a masked language modeling (MLM) objective. This model is trained on uppercase amino acids: it only works with capital letter amino acids.
[Paper](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3)
[Repo](https://github.com/agemagician/ProtTrans)

## ProtBERT-BFD
[Paper](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v4)
[Repo](https://github.com/agemagician/ProtTrans)



## AlphaFold2
Protein structure prediction

AlphaFold is an AI system developed by Google DeepMind that predicts a protein's 3D structure from its amino acid sequence
[Paper](https://www.nature.com/articles/s41586-021-03819-2)
[Repo](https://github.com/deepmind/alphafold)

## DeepFRI
Protein function prediction

DeepFRI is a structure-based protein function prediction (and functional residue identification) method using Graph Convolutional Networks with Language Model features.
[Paper](https://www.biorxiv.org/content/10.1101/786236v2)
[Repo](https://github.com/flatironinstitute/DeepFRI)

## DistilProtBERT
protein language model

A distilled protein language model used to distinguish between real proteins and their randomly shuffled counterparts
[Paper](https://academic.oup.com/bioinformatics/article/38/Supplement_2/ii95/6701995?login=true)
[Repo](https://github.com/yarongef/DistilProtBert)

## EvoDiff
protein generation

Combines evolutionary-scale data with the distinct conditioning capabilities of diffusion models for controllable protein generation in sequence space
[Paper](https://www.biorxiv.org/content/10.1101/2023.09.11.556673v1)
[Repo](https://github.com/microsoft/evodiff)

## HelixFold
protein structure prediction

An Efficient Implementation of AlphaFold2 using PaddlePaddle
[Paper](https://arxiv.org/abs/2207.05477)
[Repo](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_folding/helixfold)

## OmegaFold
protein structure prediction

OmegaFold is the first computational method to successfully predict high-resolution protein structure from a single primary sequence alone
[Repo](https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1)
[Repo](https://github.com/HeliXonProtein/OmegaFold)

## PRoBERTa
protein language model

The PRoBERTa model is fine-tuned to solve two prediction tasks, protein family memberships and protein-protein interactions.
[Paper](https://dl.acm.org/doi/10.1145/3388440.3412467)
[Repo](https://github.com/annambiar/PRoBERTa)

## ProGen
protein language model

 ProGen can learn the language of biology in order to controllably generate proteins.
[Repo](https://www.nature.com/articles/s41587-022-01618-2)
[Repo](https://github.com/salesforce/progen)

## Protein Generator
protein generation

Generate sequence-structure pairs with RoseTTAFold
[Paper](https://www.biorxiv.org/content/10.1101/2023.05.08.539766v1)
[Repo](https://github.com/RosettaCommons/protein_generator)

## ProteinMPNN
protein generation

ProteinMPNN generates highly stable sequences for designed backbones, and for native backbones, it generates sequences that are predicted to fold to the intended structures more confidently than their native sequences.
[Paper](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)
[Repo](https://github.com/dauparas/ProteinMPNN)

## ProtENN
protein function prediction
[Paper](https://www.nature.com/articles/s41587-021-01179-w)
[Repo](https://github.com/google-research/google-research/tree/master/using_dl_to_annotate_protein_universe#availability-of-trained-models)

## ProtEnT
protein language model
[Paper](https://www.biorxiv.org/content/10.1101/2023.07.15.549154v2)
[Repo](https://github.com/GrayLab/MaskedProteinEnT)



## RGN2
protein language model
[Paper](https://www.biorxiv.org/content/10.1101/2021.08.02.454840v1)
[Repo](https://github.com/aqlaboratory/rgn2)


## RoseTTaFold
protein structure prediction

RoseTTAFold Diffusion is a guided diffusion model that can be used to generate protein structures in seconds
[Paper](https://www.science.org/doi/10.1126/science.abj8754)
[Repo](https://github.com/RosettaCommons/RoseTTAFold)

## SaProt
protein language model

A structure-aware vocabulary for protein language modeling
[Paper](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v2)
[Repo](https://github.com/westlake-repl/SaProt)

## Transception
protein fitness prediction

Protein Fitness Prediction with Autoregressive Transformers and Inference-time Retrieval
[Paper](https://arxiv.org/abs/2205.13760)
[Repo](https://github.com/OATML-Markslab/Tranception)

## TransFun
protein function prediction

A method using a transformer-based protein language model and 3D-equivariant graph neural networks (EGNN) to distill information from both protein sequences and structures to predict protein function in terms of Gene Ontology (GO) terms
[Paper](https://pubmed.ncbi.nlm.nih.gov/37387145/)
[Repo](https://github.com/jianlin-cheng/TransFun)

## xTrimoPGLM
protein language model

Predicts antibody naturalness and structures, both essential to the field of antibody-based drug design
[Paper](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v3)
[Repo](https://github.com/biomap-research/xTrimoMultimer)

## REXzyme
A Translation Machine for the Generation of New-to-Nature Enzymes





