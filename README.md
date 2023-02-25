# README

Thanks for your attention. The following instructions can help you reproduce the experiments.

## Platform

Our experiments are conducted on a platform with NVIDIA GeForce RTX 3090.

## Environment

```
conda env create -f environment.yaml
```

## Running

###Data Generation
```
cd Generator/code
python generator.py --max_output_legth 30 --task_file "./task_specs/huffpost1.json" --batch_size 50
```
#### Files Definition
- ```task_specs``` : contains the instructions used to generate samples, which are composed of label descriptions wrapped by manual templates. 
- ```out```: contains the output generated samples.
### Classifier based-on Task-Conversion
- Few-shot Text Classification (Including N-way 0-shot)
```
cd GTC-FSC/code
bash train.sh
```

- Zero-shot Text Classification 
```
cd GTC-ZSC/code
bash train.sh
```
The detailed configurations can be found in the ```train.sh```. 

#### Files Definition
- ```data``` : contains the public datasets.
  
- ```G_data```: contains the generated datasets and the target label information.
