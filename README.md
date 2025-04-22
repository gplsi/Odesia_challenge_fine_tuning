# LMM_continual_fine_tunning


Documentación del repositorio utilizado para hacer Continual Pretraining con Flor 1.3B y 6.3B.

Para montar el environment se debe crear el siguiente contenedor:


```bash
## Requeriments.txt
docker run --name Pytorch_Fabric_lightning -it --net=host --gpus '"device=2,3,5,7"' -v /raid/gplsi/robiert/docker_vol/Pytorch_Fabric_lightning/:/workspace -v /raid/gplsi/NAS/GPLSI/:/workspace/NAS nvcr.io/nvidia/pytorch:24.02-py3 bash
```
Se agregan las GPUs del servidor que se van a ejecutar:
    
    ```bash
        --gpus '"device=2,3,5,7"'
    ```
Además se agregan dos volumenes:
        
        ```bash
            -v /raid/gplsi/robiert/docker_vol/Pytorch_Fabric_lightning/:/workspace
            -v /raid/gplsi/NAS/GPLSI/:/workspace/NAS
        ```
El primero hace referencia al código que se va a ejecutar, el segundo a los datos que se van a utilizar.

Por último se agrega la imagen docker que se va a utilizar:
    
        ```bash
            nvcr.io/nvidia/pytorch:24.02-py3
        ```

Para instalar los requerimientos se debe ejecutar el siguiente comando:
    
    ```bash
        pip install -r requirements.txt
    ```

Para ejecutar el código se debe ejecutar el siguiente comando:
    
    ```bash
        fabric run --node-rank=0 --accelerator=cuda --devices=2 --num-nodes=1 src/fabric_fine_tuning_full.py --devices 2
        fabric run --node-rank=0 --accelerator=cuda --devices=4 --num-nodes=1 src/fabric_fine_tuning_full.py --devices 4
    ```

Para convertir el modelo a huggingface ejecutaremos:

    ```bash
        python src/convert_fabric_to_hf_models.py --config_model /workspace/data/mock_model_config.json --checkpoint_path /workspace/models/final/lit_model.pth --devices 1 --output_dir /workspace/models/final/huggingface_model
    ```


En este caso se utiliza el framework Fabric para ejecutar el código. Se debe tener en cuenta que se debe tener instalado Fabric en el entorno de ejecución. Además se específican la cantidad de nodos, GPUs, node-rank y el archivo que se va a ejecutar.

## Ejecuciones:


### Modelo v1

- Ruta al modelo: /home/gplsi/NAS/GPLSI/odesiaChallenge/models/Llama_1B_v1/final/
- Ruta al modelo en formato HuggingFace: /home/gplsi/NAS/GPLSI/odesiaChallenge/models/Llama_1B_v1/final/huggingface_model
- WanDB: https://wandb.ai/gplsi_continual/fabric_Aitana_instruction/runs/csylzfpv

### Modelo v2

- Ruta al modelo: /home/gplsi/NAS/GPLSI/odesiaChallenge/models/Llama_3B_v1/final/
- Ruta al modelo en formato HuggingFace: /home/gplsi/NAS/GPLSI/odesiaChallenge/models/Llama_3B_v1/final/huggingface_model
- WanDB: https://wandb.ai/gplsi_continual/fabric_Aitana_instruction/runs/x7butuqs

### Modelo v3
Modelo base

- Ruta al modelo: /home/gplsi/NAS/GPLSI/odesiaChallenge/models/Llama_3B_v1/final/
- Ruta al modelo en formato HuggingFace: /home/gplsi/NAS/GPLSI/odesiaChallenge/models/Llama_3B_v2/final/huggingface_model
- WanDB: 