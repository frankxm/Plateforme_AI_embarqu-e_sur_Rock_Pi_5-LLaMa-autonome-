from rkllm.api import RKLLM
modelpath = 'Llama-2-7b-chat-hf'  # Chemin vers le modèle Llama-2-7b-chat-hf
llm = RKLLM()  # Initialisation de l'objet LLM (Large Language Model)

# Charger le modèle depuis Hugging Face
ret = llm.load_huggingface(model = modelpath)
if ret != 0:  # Si le chargement échoue
    print('Le chargement du modèle a échoué !')
    exit(ret)  # Quitter le programme avec le code d'erreur

# Construire le modèle
ret = llm.build(do_quantization=True,  # Activer la quantification pour réduire la taille et accélérer l'inférence
                optimization_level=1,  # Niveau d'optimisation (1 = optimisation de précision)
                quantized_dtype='w8a8',  # Type de quantification : 8 bits pour les poids (w) et les activations (a)
                target_platform='rk3588')  # Plateforme cible : Rockchip RK3588
if ret != 0:  # Si la construction échoue
    print('La construction du modèle a échoué !')
    exit(ret)

# Exporter le modèle au format RKLLM
ret = llm.export_rkllm("./Llama-2-7b-chat-hf.rkllm")  # Sauvegarder le modèle exporté sous le nom spécifié
if ret != 0:  # Si l'exportation échoue
    print('L'exportation du modèle a échoué !')
    exit(ret)
