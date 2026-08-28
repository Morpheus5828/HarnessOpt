Salut Claude, je travaille sur un projet qui a pour objectif de faire du rooting de cable dans des helico (de différentes tailles).
Mon idée est de partir d'une distance géodésic par exemple et décaller des masses de points présent sur le chemin afin de respecter certaines règles d'intégrations.
Le tout en utilisant de l'apprentissage par renforcement.
J'ai donc configurer une fonction reward définie par: n'avoir aucune interférence ou contact avec le maillage de l'envirronement, être à proximité de celui la entre des marges min et max. L'idée est d'être entre 1 et 10cm de l'envirronement.
Attention parfois l'envirronement peut évoluer, lors de la fusion des stl au début du cycle de rooting, certaines couleurs de maillages ont besoin que leur distance avec le cable soit plus grande. (on verra cela par la suite).
Il faut bien évidement que le cable soit lisse le plus possible et même lors du rendu finale, je suis sencé voir un cable lisser avec des rayons de courbures ne faisant pas office de cassure comme si on tordait un cable. 
Avant de rooter, il y a un phase d'analyse des points de fixations déjà existant. Si jamais tous les 25cm je n'ai pas de point de fixation, l'agent est cencé posé des crabes en s'assurant que chaque crabe soit bien parallèle à l'envirronement.

Ce que j'attend de toi et que tu regardes tout le code , que tu proposes une meilleur interface graphique, plus lisible une IHM pratique pour des NON informaticiens puissent s'en servir.
Je te laisse quartier libre sur la GUI. Concernant les agents et le problème que je t'ai défini, je veux pouvoir jouer avec Exploration et Exploitation, quitte à avoir plusieurs agent en parallèle ou même des agents qui se consacrent
à faire différentes t^ches. l'objectif est vraiment de faire un rooting LISSE et qui respecte les règles de digital mock up  (aucun clash par exemple). Et qu'ils soient aussi intéligent lors du rooting, par exemple 
de ne pas passer à travers une carré vide, rester droit le plus longtemps possible, le moins de courbure si possible. Je comte sur toi.
