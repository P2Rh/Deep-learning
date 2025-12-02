# Deep-learning

L'architecture DEVANet est conçue pour l'analyse des sentiments en fusionnant les données de trois modalités (Texte, Audio et Vision). Le processus commence par l'encodage de chaque modalité en une séquence de tokens de taille fixe : le Texte ($X_t$) utilise BERT, l'Audio ($X_a$) et la Vision ($X_v$) sont transformées via des projecteurs pour correspondre au format du texte. Le cœur du modèle est l'unité de Fusion Modale Mineure (MFU), qui utilise l'Attention Croisée où le texte ($X_t$) agit comme un guide (Query) pour extraire l'information pertinente des autres modalités ($X_a$ et $X_v$), les ajoutant de manière résiduelle à une fusion initiale. Une fois la représentation multimodale riche affinée, elle est condensée par une moyenne temporelle et passée à un classifieur simple qui prédit le score de sentiment final.

# Data
https://www.kaggle.com/datasets/reganw/cmu-mosi
