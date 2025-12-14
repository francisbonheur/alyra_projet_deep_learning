# Quickstart

```
# docker build -t alyra-dl-deploy .
# docker run -d -p 8000:8000 alyra-dl-deploy:latest
```

# Description 

## Entrainement du modèle 
Le modèle de détection de scène violente est un feature extract à partir du modèle MobileNet v2 (modèle de computer vision temps réel adapté au web). Il a été entraîné sur un corpus d’image kaggle. 
Une fois les résultats analysés, le modèle a été enregistré au format keras. Le modèle sauvegardé ainsi que le notebook d’étude est poussé sur Git

## Conception de l’API de modération d’image
Une unique API doit permettre la modération des images chargées dans la capsule temporelle. Cette API doit donc appeler tous les modèles réalisés pour la modération (images à caractère sexuelles, présentant des scènes de violence, ou autres contenus inappropriés.

Pour des raisons de performances, la génération des prédictions est réalisée de façon asynchrones. Les API suivantes ont été implémentés

 - POST /image/compliance/request
   - Création en base d’une demande de prédiction de conformité pour une image donnée au statut “pending”
   - Renvoie de l’ID de la demande sauvegardé
   - Réalisation de la prédiction en asynchrone par l’ensemble des modèles de computer vision pré-chargés
     - Appel en séquence de la méthode “predict” de chaque modèle chargé
     - Arrêt si un des modèle prédit une non-conformité
     - Le résultat de la prédiction contient le prédiction finale (0 si conforme et 1 si non conforme) ainsi que les probabilités de conformité de chaque modèle appelé
   - Sauvegarde de la prédiction en base et mise à jour du statut de la demande (“Done”)

 - GET /image/compliance/request/{id}
   - Vérification du statut de réalisation de la demande identifiée par son ID
   - Renvoie du résultat de la prédiction (la prédiction est vide si le traitement est toujours en cours)

Un endpoint permet de vérifier l’état de chargement des différents modèles de computer vision

- GET /image/compliance/healhcheck

Chaque classe responsable de charger un modèle et de réaliser une prédiction de computer vision hérité d’une classe PredictionModel et doit être annotée @prediction_model pour être chargée par un registre listant tous les modèles à exécuter. 

