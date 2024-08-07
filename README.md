# generative-biology-models

Collection of generative biology models

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/samwell/genBio)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S1H8a2i-ETLCT43mFwX2Q4yboHrtcc9X)

Refer to our [paper](https://in-vivo-group.github.io/generative-biology/) for a detailed description of these models as seen in [Protein large language models](https://in-vivo-group.github.io/generative-biology/#protein-large-language-models-prot-llms.) and [Genomic large language models](https://in-vivo-group.github.io/generative-biology/#genomic-large-language-models-gene-llms). Our focus is on prompting and generating novel sequences so we'll be mostly considering decoder-based and encoder-decoder based models.

## Engineering prompts
Case study: PoET [engineering prompt](https://www.openprotein.ai/poet-a-high-performing-protein-language-model-for-zero-shot-prediction#:~:text=false%20positive%20rate.-,Engineering%20prompts,-PoET%27s%20predictions%20can) from [OpenProtein.AI](https://www.openprotein.ai/)

## Protein large language models
- [Decoder-based models](#decoder-based-models)
  - [ProGen](#progen)
  - [ProtGPT2](#protgpt2)
  - [RITA](#rita)
  - [PoET](#poet)
  - [LM-Design](#lm-design)
  - [ZymCTRL](#zymctrl)
  - [IgLM](#iglm)

- [Encoder-decoder based models](#encoder-decoder-based-models)
  - [ProstT5](#prostt5)
  - [pAbT5](#pabt5)
  - [xTrimoPGLM](#xtrimopglm)
  - [Small-Scale protein Language Model (SS-pLM)](#ss-plm)
  - [MSA2Prot](#msa2prot)
  - [MSA-Augmenter](#msa-augmenter)
  - [Fold2Seq](#fold2seq)

## Decoder-based models

### ProGen
[Paper](https://www.nature.com/articles/s41587-022-01618-2)
[Repo](https://github.com/salesforce/progen)

ProGen, is a high capacity language model trained on the largest protein database available (~280 million samples). ProGen tackles one of the most challenging problems in science and indicates that large-scale generative modeling may unlock the potential for protein engineering to transform synthetic biology, material science, and human health. Progen demonstrates that an artificial intelligence (AI) model can learn the language of biology in order to generate proteins in a controllable fashion.

<details>
  <summary>Setup</summary>

  ```py
  # cloning the repo

  git clone https://github.com/salesforce/progen
  cd progen/progen2

  # downloading the checkpoint

  model=progen2-large
  wget -P checkpoints/${model} https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz
  tar -xvf checkpoints/${model}/${model}.tar.gz -C checkpoints/${model}/

  # setting up a virtual environment

  python3.8 -m venv .venv
  source .venv/bin/activate
  pip3 install --upgrade pip setuptools
  pip3 install -r requirements.txt

  ```
</details>

<details>
  <summary>Generating samples</summary>

  ```py
  
  python3 sample.py --model ${model} --t 0.8 --p 0.9 --max-length 1024 --num-samples 2 --context "1"
  ```
</details>

<details>
  <summary>Log likelihood</summary>

  Calculating the log-likelihood helps in assessing how well the generated protein sequences match real-world protein sequences or how likely a given sequence is under the model.

  ```py
python3 likelihood.py --model ${model} --context "1MGHGVSRPPVVTLRPAVLDDCPVLWRWRNDPETRQASVDEREIPVDTHTRWFEETLKRFDRKLFIVSADGVDAGMVRLDIQDRDAAVSVNIAPEWRGRGVGPRALGCLSREAFGPLALLRMSAVVKRENAASRIAFERAGFTVVDTGGPLLHSSKARLHVVAAIQARMGSTRLPGKVLVSIAGRPTIQRIAERLAVCQELDAVAVSTSVENRDDAIADLAAHLGLVCVRGSETDLIERLGRTAARTGADALVRITADCPLVDPALVDRVVGVWRRSAGRLEYVSNVFPPTFPDGLDVEVLSRTVLERLDREVSDPFFRESLTAYVREHPAAFEIANVEHPEDLSRLRWTMDYPEDLAFVEAVYRRLGNQGEIFGMDDLLRLLEWSPELRDLNRCREDVTVERGIRGTGYHAALRARGQAP2"

  ```
  
</details>

### ProtGPT2
[Paper](https://www.nature.com/articles/s41467-022-32007-7)
[Repo](https://huggingface.co/nferruz/ProtGPT2)

ProtGPT2 is a language model that speaks the protein language and can be used for de novo protein design and engineering. ProtGPT2 generated sequences conserve natural proteins' critical features (amino acid propensities, secondary structural content, and globularity) while exploring unseen regions of the protein space.


<details>
  <summary>Setup</summary>

  In the example below, ProtGPT2 generates sequences that follow the amino acid 'M'. Any other amino acid, oligomer, fragment, or protein of choice can be selected instead. The model      will generate the most probable sequences that follow the input. Alternatively, the input field can also be left empty and it will choose the starting tokens.
  
  ```py
  from transformers import pipeline
  protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")
  
  ```
</details>

<details>
  <summary>Generating samples</summary>

  ```py
  sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)
  for seq in sequences:
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
<summary>Compute the perplexity</summary>

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


### RITA
[Paper](https://huggingface.co/papers/2205.05789)
[Repo](https://huggingface.co/lightonai/RITA_xl)

A suite of autoregressive generative models for protein sequences, with up to 1.2 billion parameters, trained on over 280 million protein sequences belonging to the UniRef-100 database.
RITA studies how capabilities evolve with models size for autoregressive transformers in the protein domain:
RITA models were evaluated in next amino acid prediction, zero-shot fitness, and enzyme function
prediction, showing benefits from increased scale.

<details>
<summary>Setup</summary>

  ```py
  from transformers import AutoModel, AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("lightonai/RITA_xl, trust_remote_code=True")
  tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_xl")

  from transformers import pipeline
  rita_gen = pipeline('text-generation', model=model, tokenizer=tokenizer)

  ```
</details>

<details>
  <summary>Generating samples</summary>

  ```py
  sequences = rita_gen("MAB", max_length=20, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=2, eos_token_id=2)
  for seq in sequences:
      print(f"seq: {seq['generated_text'].replace(' ', '')}")
  ```
</details>

<details>
  <summary>Perplexity evaluation</summary>
  In all cases, performance is correlated with models size and RITA-XL provides the best results.
  
  ![perplexity](https://github.com/In-Vivo-Group/generative-biology-models/assets/56901167/0bab9572-6ae5-4a2d-b421-496914acee6d)


</details>

<details>
  <summary>Fitness calculation</summary>
  The ability of RITA models to predict the effects of mutations by interpreting the likelihood that the model outputs for a given protein as its fitness value was assessed. 
  <a href="https://github.com/lightonai/RITA/blob/master/compute_fitness.py">Learn more</a>
  
</details>


### PoET

[Paper](https://arxiv.org/abs/2306.06156)
[Repo](https://github.com/OpenProteinAI/PoET)

PoET is a generative model of protein families as sequences-of-sequences. It is a state-of-the-art protein language model for variant effect prediction and conditional sequence generation.

<details>
  <summary>Setup</summary>

  #### Creating an MSA
  ```py

  # from a seed sequence

  seed = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"

  # Use the Align module to create an MSA from your seed sequence

  msa = session.align.create_msa(seed.encode())
  r = msa.wait()
  msa.get_msa()

  # uploading an MSA

  f = ">101\nAAALLLPPP"
  msa = session.align.upload_msa(f.encode())
  
  

  ```

 #### Creating a prompt
 Prompt is an input that instructs the PoET model to generate the desired response. PoET uses a prompt made up of a set of related sequences that encode information about the fitness landscape of a protein of interest. These sequences may be homologs, family members, or some other grouping that represents your protein of interest.

 ```py
# generating your prompt
num_prompts = 3
prompt = msa.sample_prompt(num_ensemble_prompts=num_prompts, random_seed=42)
print(prompt)

import pandas as pd
prompt.wait()
prompt_result = []
for i in range(num_prompts):
    prompt_result.append( pd.DataFrame( list(prompt.get_prompt(i)) , columns=['name','sequence']) )

prompt.id

prompt_result

```

 #### Scoring sequences
  Scoring your sequences is a starting point for predicting the outcomes of a specific sequence or prioritizing variants for further analysis.
  PoET returns a log-likelihood score, which quantifies the model’s level of confidence in the generated sequence. The higher or less negative the score is, the more fit the sequence.

 ```py
poet = session.embedding.get_model("poet")
seqs = ["AAAGGG","LKALKA", "PGIAAA"]

poet = session.embedding.get_model('poet')

# Initiate scoring
score_job = poet.score(prompt=prompt.prompt_id, sequences=seqs )

# View your results
score_results = score_job.wait()
score_results
 ```

#### PoET single site analysis
Single site analysis using PoET scores all single substitution variants of an input sequence with a given prompt. Use this as a starting point to design single mutant or combinatorial variant libraries and predict the strength of protein activity.

```py
poet = session.embedding.get_model("poet")
sspjob = poet.single_site(prompt=prompt, sequence="AAPLAA".encode())

# Retrieve and view your results

ssp_results = sspjob.wait()
ssp_results

# Access specific sites
ssp_results[b'A1R']

```

 
</details>

<details>
  <summary>Generating samples</summary>
  Use the PoET model to generate de novo sequences conditioned on the sequence context provided by a prompt. Use this as a starting point for generating a diverse library without 
  existing experimental data.

  `prompt` : Uses a prompt from an align workflow to condition Poet model.
  
  `num_samples` : Indicates the number of samples to generate. The default is 100.
  
  `temperature` : The temperature for sampling. Higher values produce more random outputs. The default is 1.0.
  
  `topk` : The number of top-k residues to consider during sampling. The default is None.
  
  `topp` : The cumulative probability threshold for top-p sampling. The default is None.
  
  `max_length` : The maximum length of generated proteins. The default is 1000.
  
  `seed` : Seed for random number generation. The default is a random number.

  
  ```py
  # Generating 10 samples
  poet = session.embedding.get_model("poet")
  genjob = poet.generate(prompt=prompt, num_samples=100) #prompt_id from your previous prompt job

  # View your results once the job is complete
  gen_results = genjob.wait()
  gen_results
  

  ```

  
  
</details>


### LM-Design

[Paper](https://arxiv.org/abs/2302.01649)
[Repo](https://github.com/BytedProtein/ByProt)

LM-Design is a generic approach to reprogramming sequence-based protein language models (pLMs), that have learned massive sequential evolutionary knowledge from the universe of natural protein sequences, to acquire an immediate capability to design preferable protein sequences for given folds. LM-Design demonstrates that language models are strong structure-based protein designers.

<details>
  <summary>Setup</summary>

  ```py

  # clone project
git clone --recursive https://url/to/this/repo/ByProt.git
cd ByProt

# create conda virtual environment
env_name=ByProt

conda create -n ${env_name} python=3.7 pip
conda activate ${env_name}

# automatically install everything else
bash install.sh

 ```

```py
from byprot.utils.config import compose_config as Cfg
from byprot.tasks.fixedbb.designer import Designer

# 1. instantialize designer
exp_path = "/root/research/projects/ByProt/run/logs/fixedbb/cath_4.2/lm_design_esm2_650m"
cfg = Cfg(
    cuda=True,
    generator=Cfg(
        max_iter=5,
        strategy='denoise', 
        temperature=0,
        eval_sc=False,  
    )
)
designer = Designer(experiment_path=exp_path, cfg=cfg)

# 2. load structure from pdb file
pdb_path = "/root/research/projects/ByProt/data/3uat_variants/3uat_GK.pdb"
designer.set_structure(pdb_path)
```

</details>

<details>
  <summary>Generating samples</summary>

  ```py
  # 3. generate sequence from the given structure
designer.generate()
# you can override generator arguments by passing generator_args, e.g.,
designer.generate(
    generator_args={
        'max_iter': 5, 
        'temperature': 0.1,
    }
)

# 4. calculate evaluation metircs
designer.calculate_metrics()
## prediction: LNYTRPVIILGPFKDRMNDDLLSEMPDKFGSCVPHTTRPKREYEIDGRDYHFVSSREEMEKDIQNHEFIEAGEYNDNLYGTSIESVREVAMEGKHCILDVSGNAIQRLIKADLYPIAIFIRPRSVENVREMNKRLTEEQAKEIFERAQELEEEFMKYFTAIVEGDTFEEIYNQVKSIIEEESG
## recovery: 0.7595628415300546
  ```

  
</details>

### ZymCTRL

[Paper](https://www.biorxiv.org/content/10.1101/2024.05.03.592223v1)
[Repo](https://huggingface.co/AI4PD/ZymCTRL)

ZymCTRL (Enzyme Control) is a conditional language model for the generation of artificial functional enzymes. It was trained on the UniProt database of sequences containing (Enzyme Commission) EC annotations, comprising over 37 M sequences. Given a user-defined Enzymatic Commission (EC) number, the model generates protein sequences that fulfil that catalytic reaction. The generated sequences are ordered, globular, and distant to natural ones, while their intended catalytic properties match those defined by users.

<details>
<summary>Setup, sample generation and perplexity</summary>

Given that generation runs so fast, it is recommended that hundreds or thousands are generated and then only picking the best 5% or less. With the script below, that would mean picking only those that finish in '_0.fasta'. Good perplexity values for this model so be below 1.75-1.5.

  ```py
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import math

def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids,model,tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def main(label, model,special_tokens,device,tokenizer):
    # Generating sequences
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids, 
        top_k=9, #tbd
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
           do_sample=True,
           num_return_sequences=20) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    
    # Check sequence sanity, ensure sequences are not-truncated.
    # The model will truncate sequences longer than the specified max_length (1024 above). We want to avoid those sequences.
    new_outputs = [ output for output in outputs if output[-1] == 0]
    if not new_outputs:
        print("not enough sequences with short lengths!!")

    # Compute perplexity for every generated sequence in the batch
    ppls = [(tokenizer.decode(output), calculatePerplexity(output, model, tokenizer)) for output in new_outputs ]

    # Sort the batch by perplexity, the lower the better
    ppls.sort(key=lambda i:i[1]) # duplicated sequences?

    # Final dictionary with the results
    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return sequences

if __name__=='__main__':
    device = torch.device("cuda") # Replace with 'cpu' if you don't have a GPU - but it will be slow
    print('Reading pretrained model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('/path/to/zymCTRL/') # change to ZymCTRL location
    model = GPT2LMHeadModel.from_pretrained('/path/to/zymCTRL').to(device) # change to ZymCTRL location
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    # change to the appropriate EC classes
    labels=['3.5.5.1'] # nitrilases. You can put as many labels as you want.

    for label in tqdm(labels):
        # We'll run 100 batches per label. 20 sequences will be generated per batch.
        for i in range(0,100): 
            sequences = main(label, model, special_tokens, device, tokenizer)
            for key,value in sequences.items():
                for index, val in enumerate(value):
                    # Sequences will be saved with the name of the label followed by the batch index,
                    # and the order of the sequence in that batch.           
                    fn = open(f"/path/to/folder/{label}_{i}_{index}.fasta", "w")
                    fn.write(f'>{label}_{i}_{index}\t{val[1]}\n{val[0]}')
                    fn.close()
  ```
</details>




### IgLM

[Paper](https://www.biorxiv.org/content/10.1101/2021.12.13.472419v1)
[Repo](https://github.com/Graylab/IgLM)

Immunoglobulin Language Model (IgLM), a deep generative language model for generating synthetic libraries by re-designing variable-length spans of antibody sequences. IgLM formulates anti-body design as an autoregressive sequence generation task based on text-infilling in natural language. We trained IgLM on approximately 558M antibody heavy- and light-chain variable sequences, conditioning on each sequence’s chain type and species-of-origin.

<details>
  <summary>Setup & sequence generation</summary>

  ```py
  pip install iglm

  ```

### Full antibody sequence generation

```py
from iglm import IgLM

iglm = IgLM()

prompt_sequence = "EVQ"
chain_token = "[HEAVY]"
species_token = "[HUMAN]"
num_seqs = 100

generated_seqs = iglm.generate(
    chain_token,
    species_token,
    prompt_sequence=prompt_sequence,
    num_to_generate=num_seqs,
)
```
</details>

<details>
  <summary>Sequence evaluation</summary>

  IgLM can be used to calculate the log likelihood of a sequence given a chain type and species-of-origin.

Full sequence log likelihood calculation:

```py
import math
from iglm import IgLM

iglm = IgLM()

sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATISADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVS"
chain_token = "[HEAVY]"
species_token = "[HUMAN]"

log_likelihood = iglm.log_likelihood(
    sequence,
    chain_token,
    species_token,
)
perplexity = math.exp(-log_likelihood)
```
</details>



## Encoder-decoder based models

### ProstT5

Details about ProstT5 go here.

### pAbT5

Details about pAbT5 go here.

### xTrimoPGLM

Details about xTrimoPGLM go here.

### Small-Scale protein Language Model (SS-pLM)

Details about SS-pLM go here.

### MSA2Prot

Details about MSA2Prot go here.

### MSA-Augmenter

Details about MSA-Augmenter go here.

### Fold2Seq

Details about Fold2Seq go here.


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





