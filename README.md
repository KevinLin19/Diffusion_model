Pour pouvoir tester la version 1 du ControlNet, il faut créer un dossier 'data' où il faudra mettre le dataset mnist comme suit : 

-> data
    -> mnist
        -> train
            -> images
                -> *.png
        -> test
            -> images
                -> *.png

Il faudra aussi télécharger les poids d'un modèle pré-entrainé dans un fichier que vous devrez appeler 'ddpm_ckpt_pre_trained.pth'. Tout cela sera dans un dossier appelé 'DDPM_trained'.

