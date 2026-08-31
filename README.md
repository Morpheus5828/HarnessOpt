# HarnessOpt

Cheminement assisté de harnais électriques dans des cellules d'hélicoptère.

Le principe : chercher un premier chemin entre deux points dans l'espace libre
de la maquette numérique, puis faire déplacer les points de ce chemin par une
équipe d'agents d'apprentissage par renforcement, jusqu'à obtenir un tracé qui
respecte les règles d'intégration (aucune interférence, distances tenues,
rayons de cintrage admissibles, fixations tous les 250 mm).

---

## Démarrer

```bash
pip install -r requirements.txt
python main.py
```

Le dossier de travail est choisi automatiquement (`%LOCALAPPDATA%\HarnessOpt`
sous Windows, `~/.cache/harnessopt` ailleurs). Pour le forcer :

```bash
HARNESSOPT_CACHE=/chemin/de/mon/cache python main.py
```

L'export CATIA (`core/catia_handler.py`) nécessite Windows, CATIA lancé et
pywin32. Sans eux, tout le reste fonctionne : il suffit de travailler à partir
d'un dossier de STL déjà exportés. Le connecteur s'importe sur toutes les
plateformes et ne signale l'absence de CATIA qu'au moment où on l'appelle,
avec un message distinct selon la cause (pywin32 absent, CATIA injoignable,
macro en échec).

---

## L'interface

Un assistant en quatre étapes, qui se déverrouillent au fur et à mesure.

| Étape | Ce qu'on y fait |
|---|---|
| **1. Projet** | Désigner le dossier de STL (ou lancer l'export CATIA), assembler la maquette, voir la répartition des pièces par famille. |
| **2. Règles** | Cocher les règles à appliquer, puis régler diamètre du toron, rayon de cintrage, distances mini/maxi, distances renforcées par famille, pas entre fixations. |
| **3. Cheminement** | Poser départ et arrivée, choisir une équipe d'agents et le réglage exploration/exploitation, lancer, suivre en direct (conformité, conseils, agents, courbes). |
| **4. Rapport** | Lire le verdict règle par règle, exporter en STL, CSV ou JSON, réinsérer le faisceau dans le document CATIA ouvert. |

Tout est exprimé en unités physiques et en vocabulaire métier. Les
hyperparamètres d'algorithme ne sont pas exposés par défaut : le curseur
**Exploration ↔ Exploitation** et le choix d'équipe les pilotent. Ils restent
accessibles sous « Réglages avancés » pour qui veut y toucher.

Une étape est accessible ou non selon **l'état réel du projet**, jamais selon
les écrans déjà visités : *Règles* s'ouvre dès que la maquette est chargée,
*Cheminement* dès que les règles sont cohérentes, *Rapport* dès qu'un
cheminement a démarré. Si une condition cesse d'être remplie — des règles
rendues incohérentes, par exemple — l'étape se referme et l'application dit
laquelle des conditions manque. On avance soit par le bouton en bas de chaque
écran, soit en cliquant directement le numéro d'étape en haut.

---

## Les règles d'intégration

Elles sont définies une seule fois, dans `core/routing_rules.py`, et servent
**à la fois** de fonction de récompense pour les agents et de grille de
contrôle affichée à l'écran. C'est volontaire : sans définition partagée, un
agent peut converger vers une route que l'intégrateur refuserait.

| Règle | Gravité | Mesure |
|---|---|---|
| Aucune interférence avec le DMU | rédhibitoire | traversées de faces + points dans la matière |
| Distance minimale respectée | rédhibitoire | distance au maillage, exigence par famille de pièce |
| Rayon de cintrage admissible | rédhibitoire | rayon de congé réalisable, comparé à 6 × Ø |
| Câble maintenu à portée de la structure | majeure | distance maximale |
| Pas de traversée dans le vide | majeure | longueur de la portion sans appui possible |
| Une fixation au moins tous les 250 mm | majeure | plus grand écart entre supports |
| Crabes posés à plat | majeure | écart angulaire embase / structure |
| Tracé rectiligne sur la majeure partie | qualité | part de longueur parcourue en ligne droite |

### Choisir les règles à appliquer

Chaque règle du tableau ci-dessus porte une case à cocher en tête de la page
*Règles*. La décocher la retire **réellement** du problème, et pas seulement du
rapport : elle cesse d'être évaluée, cesse de compter dans le classement des
agents, et cesse de peser sur la récompense. C'est cette dernière propriété qui
importe — une règle qui disparaît de l'écran tout en continuant à tirer sur les
agents produirait un comportement incompréhensible.

Une famille de récompense n'est neutralisée que si *toutes* les règles qui s'y
rattachent sont décochées. Décocher la seule distance maximale ne supprime donc
pas la pression qui maintient le câble à distance des pièces, puisque la
distance minimale, elle, reste demandée.

Décocher une règle rédhibitoire — « aucune interférence », typiquement — est
possible pour une étude exploratoire, mais l'application prévient au moment du
clic que la route obtenue pourra ne pas être livrable.

Le catalogue des règles vit dans `RULE_CATALOG` (`core/routing_rules.py`) :
ajouter une règle à ce tuple suffit à la faire apparaître à l'écran avec sa case
à cocher, son libellé bilingue et sa gravité.

### Les zigzags

Un zigzag n'est pas une question de courbure totale, et c'est pourquoi il a son
terme propre dans la récompense de **tous** les agents. Un arc de cercle
régulier accumule beaucoup de courbure sans jamais osciller ; à l'inverse, une
succession de petits virages alternés totalise peu de courbure tout en donnant
un câble visuellement inacceptable. Ni le rayon de cintrage, qui regarde le
virage local, ni la part de tracé rectiligne ne distinguent ces deux cas.

La mesure compte les **inversions du sens de virage** — produit scalaire négatif
entre binormales consécutives — et retient l'amplitude du plus petit des deux
virages en cause : une oscillation minuscule entre deux grands virages reste une
petite faute, deux grands virages opposés forment un vrai zigzag.

Aucun rôle ne descend sous le poids de référence, pas même l'éclaireur qui
néglige par ailleurs le lissage, et le terme ne dépend d'aucune case à cocher.
Mesuré sur 165 itérations, il fait passer l'éclaireur de 417° à 202°
d'oscillation cumulée, et le contrôleur d'écarts de 317° à 246°.

### Le rayon de cintrage

Le lissage était auparavant jugé sur le cosinus de l'angle entre deux segments
consécutifs. Ce critère dépend de l'espacement des points : comme le
raffinement adaptatif en insère et en supprime en cours de route, un même
cosinus correspondait à des rayons physiques très différents.

On mesure désormais le **rayon de congé réalisable** au sommet :

```
R = (min(L_entrant, L_sortant) / 2) / tan(theta / 2)
```

* portion droite → rayon infini ;
* courbe finement échantillonnée → converge vers le rayon de courbure réel ;
* coude franc entre deux longues lignes droites → donne le plus grand congé que
  la place disponible autorise, ce qui est exactement la question que se pose
  l'intégrateur.

La grandeur est en millimètres, comparable directement au rayon admissible du
toron, et **invariante au rééchantillonnage** : raffiner un tracé ne change
plus sa note.

### Distances différenciées par couleur

L'étape de fusion produit une table « face → famille DMU »
(`fusion/face_families.npz`). La distance minimale exigée devient alors propre
à la pièce réellement survolée : 70 mm au-dessus d'une hydraulique haute
pression, 20 mm le long d'une ligne d'air chaud, 10 mm ailleurs. Sans cette
table, une distance uniforme s'applique.

---

## Le chemin de départ

### Deux façons de relier les points

L'application propose **la recherche dans l'espace libre** (par défaut) et
**le chemin de surface**. Elles répondent à deux besoins différents.

### Le chemin de surface, et pourquoi il ne marchait pas

`PolyData.geodesic` cherche un chemin d'**arêtes** entre deux sommets. Une
maquette DMU est la fusion de centaines de pièces **disjointes** : entre deux
pièces séparées il n'existe aucune arête, donc aucun chemin. La fonction
échouait systématiquement et le code la remplaçait **sans le dire** par une
corde tendue. Choisir « le long de la surface » donnait une ligne droite.

`core/surface_path.py` construit le graphe qui manquait :

* les **arêtes du maillage**, pondérées par leur longueur — à l'intérieur d'une
  pièce, le plus court chemin y *est* la géodésique discrète ;
* des **ponts** entre pièces voisines, pondérés par leur longueur multipliée
  par une pénalité (6 par défaut). C'est cette pénalité qui fait la différence
  entre longer la structure et couper au plus court : sauter coûte six fois le
  prix du même déplacement le long d'une surface, donc le chemin ne saute que
  là où il n'a pas le choix.

Les ponts sont cherchés **par composante**, pas par plus proches voisins : sur
une pièce un tant soit peu dense, les huit voisins les plus proches d'un sommet
sont tous sur la même pièce, et aucun pont ne serait jamais créé.

Mesuré sur une sphère (maillage connexe) : rapport longueur/corde de 1,57,
aucun saut, distance moyenne au maillage nulle — c'est bien un arc de grand
cercle. Sur un chapelet de huit pièces : 79 % du trajet le long des surfaces,
sept sauts. Avec une pénalité de 1 au lieu de 6, la part sur surface tombe à
61 % : le réglage agit comme prévu.

Le graphe est restreint à un corridor ellipsoïdal autour de A–B. Sur une
maquette de 128 000 sommets, cela divise par dix-huit le temps d'un trajet
court sans changer d'un millimètre le chemin trouvé.

Un chemin de surface **colle à la structure** : c'est sa définition, et les
agents devront l'en décoller. Si vous voulez partir directement dans la bande
de distance visée, prenez la recherche dans l'espace libre.

### A\* pondéré, dont le glouton est un cas particulier

La recherche utilise `f = g + w · h`. `w = 1` donne A\* ; `w` grand rend le
terme `g` négligeable et l'on retrouve la **recherche gloutonne** (*greedy
best-first search*). Un seul paramètre couvre les deux.

Mais le coût `g` porte ici la préférence pour la bande de distance **et** la
pénalité de changement de direction. Une recherche gloutonne ne classe que sur
`h` : ces deux règles lui sont invisibles. Mesuré sur une diagonale en espace
libre, où la grille offre de nombreux escaliers de même longueur :

| stratégie | pénalité de virage | virage total | cellules explorées |
|---|---|---|---|
| Rapide (glouton, w = 12) | 0 → 6 | 135° → **135°** *(sans effet)* | 56 |
| **Équilibré (w = 1,4)** | 0 → 6 | 1125° → **45°** | 59 |
| Meilleur chemin (A\*, w = 1) | 0 → 6 | 345° → 132° | 4714 |

Le glouton est le plus rapide et reste utile sur une grande maquette, mais il
ne sait pas produire de longues lignes droites. « Équilibré » est le défaut :
ici il fait mieux qu'A\* pour 80 fois moins de calcul, la légère gourmandise
départageant les chemins que le coût seul laisse à égalité.

### Ce que la recherche garantit

* **Marge de sécurité liée à la grille.** Le champ de distance est mesuré entre
  centres de cellules ; la surface réelle peut être plus proche d'au plus une
  demi-diagonale. Cette marge est ajoutée à la distance exigée, sans quoi le
  chemin frôle les pièces alors que la grille le croit dégagé.
* **Résolution déduite de la marge visée.** Une cellule plus grande que la
  distance minimale empêche de longer la structure d'aussi près.
* **Vérification contre le maillage réel.** La distance minimale du chemin
  final est mesurée par requête de proximité, pas déduite de la grille, et
  rapportée telle quelle.
* **Aucun repli silencieux.** En cas d'échec, le lancement s'interrompt avec la
  raison : point dans la matière, marge irréalisable, passage introuvable.

La géodésique reste proposée dans l'interface, et prévient désormais quand elle
se réduit à une ligne droite.

---

## La vue 3D

Elle est incrustée dans l'écran *Cheminement* : glisser pour tourner, molette
pour zoomer, clic droit pour déplacer. « Ouvrir en grand » ouvre en plus une
fenêtre VTK native, réellement interactive.

Le point important est architectural. **Tout ce qui touche à VTK vit dans un
fil de rendu dédié**, qui possède le plotter, reçoit des ordres par une file et
renvoie des images. L'interface se contente d'afficher la dernière image reçue.
Aucun appel 3D n'a lieu sur le fil de Tk.

Ce n'est pas une élégance gratuite : construire un `pyvista.Plotter` demande
**13,6 s** sur une maquette de 800 000 triangles, mesuré. La version précédente
le faisait sur le fil de l'interface, qui restait donc figée d'autant, puis
n'affichait jamais la fenêtre ainsi construite — le cadre réservé à la 3D
restait vide sur toutes les plateformes.

---

## Suivre et débloquer une session

### Les courbes

Quatre courbes, une par agent, avec la couleur de sa trajectoire dans la vue 3D :
la **récompense** — ce que l'agent maximise réellement, seule à dire si
l'apprentissage progresse ou piétine — puis trois grandeurs physiques
(interférences, distance au DMU, rayon de cintrage) avec leur limite en
pointillés, qui disent si la route est livrable.

Les deux registres sont nécessaires. Une récompense qui monte pendant qu'un
rayon de cintrage stagne sous la limite signale une pondération mal réglée : ni
l'une ni l'autre des courbes ne le montre seule.

L'abscisse porte le **numéro d'itération**. Passé quatre cents relevés,
l'historique est décimé d'un facteur deux plutôt que tronqué : la courbe couvre
toujours toute la session, début compris, au lieu d'afficher éternellement
« 0 à 400 » alors que les agents en sont à plusieurs milliers d'itérations.

### Les conseils

Un agent qui n'arrive pas à respecter une règle ne le dit pas : il continue,
et rien ne distingue « c'est long » de « c'est impossible ». L'onglet
**Conseils** fait la différence, et la formule en termes actionnables — valeur
mesurée, valeur proposée, réglage exact, bouton pour l'appliquer.

Trois garde-fous le gouvernent :

* **Rien avant d'avoir cherché.** Les conseils n'apparaissent qu'après un
  nombre minimal d'itérations *et* une stagnation avérée du meilleur score.
  Sinon l'application inciterait à baisser les exigences au premier obstacle.
* **Jamais de relâchement sur les clashs.** Une route qui traverse la structure
  n'est pas une route ; le conseil porte alors sur la recherche — plus
  d'exploration, plus de points, points de départ mal placés — et ne propose
  aucun réglage. C'est la seule règle traitée ainsi.
* **Jamais de valeur intenable.** En deçà de 3 × Ø de rayon de cintrage ou d'un
  millimètre de distance, aucun bouton n'est proposé : la route serait déclarée
  conforme sans être posable. Le conseil dit alors de revoir le passage.

Le réglage n'est pas appliqué à chaud. Les agents ont recopié les règles au
démarrage ; n'en changer qu'une partie donnerait un mélange incohérent entre ce
qui récompense et ce qui est mesuré. Le bouton écrit dans l'étape *Règles* et
invite à relancer.

### Les fixations déjà montées

Si un dossier de modèles de fixations est indiqué à l'étape *Règles*,
l'application les recale sur la maquette par ICP avant tout cheminement. Le
bandeau en haut de l'écran *Cheminement* annonce **combien de modèles ont été
examinés et combien reconnus**, puis liste les passages imposés des peignes :
le segment `p_in` → `p_out` par lequel le câble doit traverser chaque encoche.

Ces fixations sont **dessinées dans la vue 3D** : un repère gris par fixation
reconnue, une bille verte à l'entrée de chaque encoche, une rouge à la sortie,
et le segment jaune que le câble doit emprunter. Rien ne les dessinait
auparavant — le viewer savait masquer les acteurs `clamp_`, mais aucun n'était
jamais créé, et l'utilisateur n'avait aucun moyen de vérifier ce que le scan
avait reconnu.

Une fois le scan terminé et les passages dessinés, l'application **pose la
question** : « faut-il faire passer le faisceau par ces fixations ? ». Elle
arrive après l'affichage, pas avant : l'utilisateur répond en voyant ce dont on
parle, plutôt que sur une liste de coordonnées. Sa réponse se retrouve sur
l'interrupteur *Emprunter les fixations existantes* de la page, et y reste
mémorisée. Sans passage détecté, aucune question n'est posée.

**Répondre oui contraint deux choses.** Le chemin de départ est découpé en
tronçons par les couples entrée/sortie, ordonnés le long de A→B et orientés
dans le sens de la marche ; la traversée de l'encoche elle-même est une ligne
droite, y lancer une recherche de chemin ferait contourner le peigne au lieu
de passer dedans. Et surtout, **les agents y sont maintenus** : à chaque
itération le point le plus proche de chaque passage est ramené exactement
dessus, puis retiré des points que l'agent peut déplacer.

Cette dernière contrainte n'est pas un excès de prudence. L'agent dispose déjà
d'une attraction par récompense vers les fixations existantes ; mesurée sur une
vraie boucle, elle ne suffit pas. Après deux cents itérations, le câble s'écarte
de **220 à 350 mm** des encoches — une encoche de peigne ne se négocie pas à
cette distance. Avec l'épinglage, l'écart mesuré est de **0,0 mm** sur tous les
passages, pour tous les agents.

Répondre non annule tout : ni découpage du trajet, ni épinglage, ni même
l'attraction par récompense — la liste des fixations n'est pas transmise aux
agents. Un refus doit être un vrai refus.

Le détecteur repose sur Open3D. Sans lui, le scan ne plante pas : il dit
pourquoi il n'a pas eu lieu, et le cheminement continue sans fixations
préexistantes.

### Les crabes posés par les agents

Les crabes ne sont pas ajoutés après coup : `compute_crabes` tourne **à chaque
itération**, à l'intérieur de la boucle d'apprentissage. Leur position pèse sur
la récompense (`R_crabe`, `R_fixation`) et alimente le rapport de conformité,
si bien que la trajectoire est déplacée pour aller là où l'on peut réellement
fixer. Ils sont désormais dessinés au fil du calcul : une bille dorée sur le
câble, reliée à son pied sur la structure.

Un point à connaître : **sans modèle de crabe chargeable, aucun crabe n'est
posé**, et la règle du pas entre fixations ne peut jamais passer. C'est
délibéré — poser une fixation sans vérifier qu'elle tient sur la structure et
qu'elle n'entre en collision avec rien serait faux. L'onglet *Conseils* le
signale dès le lancement plutôt que de laisser les agents tourner pour rien.

---

## Les agents

### Rôles

Cinq spécialités, décrites dans `core/orchestrator.py`. Chacune pondère
différemment les mêmes règles ; aucune n'en débranche aucune.

| Rôle | Ce qu'il cherche | Algorithme |
|---|---|---|
| **Éclaireur** | un passage praticable, quitte à être approximatif | TD3 |
| **Contrôleur d'écarts** | supprimer interférences et distances insuffisantes | TD3 |
| **Lisseur** | élargir les rayons de cintrage | SAC |
| **Rectifieur** | allonger les lignes droites, réduire le nombre de coudes | TD3 récurrent (BiGRU) |
| **Poseur de crabes** | faire passer le câble là où on peut le fixer | TD3 récurrent (BiGRU) |

Compositions proposées : *Découverte* (beaucoup d'éclaireurs), *Équilibrée*
(un par spécialité), *Finition* (lissage et fixations), *Mise en conformité*
(interférences et distances d'abord).

### Exploration ↔ exploitation

Un curseur unique, de 0 à 1, pilote six réglages de façon cohérente :

| | Exploitation (0) | Exploration (1) |
|---|---|---|
| Bruit d'exploration | 0,10 | 0,90 |
| Bruit résiduel | 0 (fige) | 0,25 |
| Amplitude des pas | × 0,5 | × 1,8 |
| Inertie | 0,85 | 0,25 |
| Intervalle entre échanges | 25 itérations | 250 itérations |
| Part de l'équipe rappelée | 50 % | 0 % |

Monter le bruit sans allonger la patience ni espacer les échanges revient à
agiter les agents sans leur laisser le temps d'aboutir : c'est pourquoi ces six
valeurs bougent ensemble. **Le curseur se déplace en cours de calcul.**

### Échanges entre agents

Régulièrement, les routes sont classées avec le score DMU (ordre
lexicographique : interférences → distances → cintrage → fixations → portées
libres → rectitude → tortuosité → longueur). Les agents en retard repartent de
la meilleure route trouvée, avec des réglages perturbés — c'est du
*Population-Based Training*.

Migrer transmet une **route**, pas un rôle : l'agent qui repart de la meilleure
solution garde ses propres poids et continue sa spécialité. La diversité de
l'équipe est donc préservée sans protection particulière ; c'est le curseur qui
dose la convergence.

### Curriculum

L'équipe suit les étapes du chantier, déduites de l'état de la meilleure route :

```
Recherche d'un passage → Mise aux distances → Pose des fixations → Lissage final
```

Le rôle attendu à l'étape en cours travaille à pleine puissance, les autres
lèvent légèrement le pied. Inutile de polir une route qui traverse encore une
cloison.

---

## Le connecteur CATIA

Le dialogue avec CATIA passe par une macro VBScript écrite à la volée puis
exécutée via `SystemService.ExecuteScript`.

**Les macros doivent être construites avec des f-strings brutes** (`rf"..."`).
VBScript n'a aucun caractère d'échappement dans ses littéraux : un antislash
doit arriver tel quel. Avec une f-string ordinaire, Python les consomme avant
l'écriture du fichier, et la ligne qui assainit les noms de pièces devient un
`Replace` de chaîne vide. Une pièce dont le nom CATIA contient un antislash
produit alors un chemin vers un sous-dossier inexistant : son export échoue
sans bruit sous `On Error Resume Next`, et la pièce manque dans la maquette
sans qu'aucun message ne le signale.

`build_export_macro` et `build_import_macro` sont des fonctions pures, isolées
du dialogue COM pour être vérifiables sans Windows — `tests/test_catia_handler.py`
couvre ce cas de régression.

---

## Organisation du code

```
main.py                     point d'entrée
config.py                   constantes historiques (distances de référence)
core/
  paths.py                  dossiers de travail, réglages persistants
  geometry_metrics.py       longueurs, courbure, rectitude, portées libres
  routing_rules.py          règles d'intégration + rapport de conformité
  diagnostics.py            conseils quand la convergence bloque
  fixation_scan.py          fixations existantes et passages imposés
  reward_terms.py           traduction des règles en signal d'apprentissage
  path_planner.py           recherche du chemin de départ dans l'espace libre
  surface_path.py           chemin le long de la surface, sauts entre pièces compris
  orchestrator.py           rôles, curseur exploration/exploitation, migrations
  agent_team.py             fabrique des réseaux + superviseur d'équipe
  agent_worker.py           boucle d'optimisation d'un agent
  mesh_processor.py         extraction et fusion du DMU (processus séparé)
  mesh_model.py             classification des pièces par couleur
  catia_handler.py          export STL et réimport du faisceau via macros CATIA
  agent/                    réseaux (acteur, critique), mémoires, outils géométriques
  path_managment/           détection des fixations existantes (ICP Open3D)
controller/
  app_controller.py         liaison interface ↔ moteur
ui/
  app_window.py             fenêtre principale (assistant 4 étapes)
  theme.py  i18n.py         charte graphique, traductions FR/EN/DE/ES
  charts.py                 courbes (récompense + grandeurs physiques)
  viewer3d.py               vue 3D incrustée (fil de rendu dédié)
  widgets/  pages/          composants et écrans
tests/                      312 tests hors interface, 133 tests d'interface
```

`core/geometry_metrics.py`, `core/routing_rules.py`, `core/reward_terms.py` et
`core/orchestrator.py` ne dépendent que de numpy : ils sont testables sans
PyTorch, sans maillage et sans écran.

### Fichiers hérités

`ui/main_window.py`, `ui/pages/extraction_view.py`, `ui/pages/agent_view.py`,
`controller/controller.py` et `core/controller/controller.py` sont l'ancienne
interface et son contrôleur. Ils ne sont plus appelés par `main.py`. Les deux
contrôleurs sont deux copies quasi identiques du même fichier. Ils sont
conservés pour référence — à supprimer quand la nouvelle interface vous
convient.

`core/sphere_generation.py`, `core/tools.py`, `core/visualize.py`,
`core/mesh_fusion.py`, `core/HS9019.py`, `core/smooth.py`,
`core/path_managment/fixation.py` et `core/path_managment/smooth.py`
appartiennent à une approche antérieure par graphe et ne sont importés par
aucun chemin actif.

---

## Tests

```bash
python -m pytest tests/ -q                       # règles, géométrie, agents
# interface, vue 3D, courbes :
xvfb-run -a python -m pytest tests/test_ui.py tests/test_viewer3d.py tests/test_charts.py
```

Les tests d'interface s'ignorent d'eux-mêmes si tkinter, customtkinter ou un
serveur graphique manquent.

`tests/test_viewer3d.py` fait tourner le vrai fil de rendu sur un plotter
factice. Il vérifie notamment que `Viewer3D.start()` rend la main en quelques
millisecondes même quand la construction du contexte 3D traîne : c'est la
garantie de non-régression du gel décrit plus haut.
