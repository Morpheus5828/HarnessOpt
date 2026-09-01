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

#### Longer la structure sans la raser

Un chemin de surface, par définition, passe **sur** le maillage. Pris tel quel
comme point de départ, il est donc en interférence sur toute sa longueur : les
agents démarrent en clash et dépensent leurs premières centaines d'itérations à
s'extraire d'une situation que le tracé initial leur a créée. Compter sur la
seule pénalité de distance pour les en sortir, c'est leur faire payer un défaut
qui n'est pas le leur.

Le tracé est donc **décollé** avant d'être rendu. Chaque point est projeté sur
la face la plus proche puis repoussé le long de sa normale jusqu'à la distance
visée ; l'opération est répétée, car déplacer un point peut lui donner un
nouveau plus proche voisin. La marge réellement obtenue est ensuite **remesurée
sur le maillage** et annoncée dans le bandeau d'état : on ne se contente pas de
la viser.

La cible est `distance mini + 25 % de la bande` : à l'intérieur du domaine
autorisé, mais pas collée à sa limite basse, où le moindre déplacement d'un
agent la ferait franchir. Mesuré sur une sphère, les cibles de 5, 20 et 60 mm
sont atteintes à moins d'un micron ; sur une maquette de pièces disjointes, le
tracé de départ passe de « en interférence » à « conforme dès l'itération 0 ».

Ce décalage ne change pas la **forme** du chemin — il suit toujours la
structure. Si vous voulez au contraire ignorer la surface et partir droit dans
la bande de distance, prenez la recherche dans l'espace libre.

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

**Une seule vue, et c'est une vraie fenêtre VTK.** L'application en tenait deux :
une image incrustée dans la page *Cheminement*, produite par capture d'écran
d'un plotter hors écran, et une fenêtre détachée optionnelle. L'incrustée
coûtait cher pour ce qu'elle montrait — une image figée, une émulation maison
de l'orbite, du panoramique et du zoom, et un pipeline de capture qui se
dispute le pilote graphique avec la fenêtre. Elle est supprimée ; la page ne
garde qu'un bandeau d'état, et toute la hauteur gagnée revient aux onglets.

Le point important reste architectural. **Tout ce qui touche à VTK vit dans un
fil de rendu dédié**, qui possède le plotter, reçoit des ordres par une file et
pompe les évènements de la fenêtre. Aucun appel 3D n'a lieu sur le fil de Tk.

Ce n'est pas une élégance gratuite : construire un `pyvista.Plotter` demande
**13,6 s** sur une maquette de 800 000 triangles, mesuré. La version d'origine
le faisait sur le fil de l'interface, qui restait donc figée d'autant.

La scène est décrite par des **recettes** — géométrie et style — pas par des
acteurs VTK. C'est ce qui permet de fermer la fenêtre, de la rouvrir, et d'y
retrouver la scène intacte, sans jamais partager d'objet VTK entre deux fils.
Les ordres d'affichage sont donc acceptés même fenêtre fermée : le contrôleur
n'a pas à savoir si quelqu'un regarde.

### Fermer la fenêtre sans noyer la console

`Plotter.update()` appelle `render()` **sans rien vérifier**. Une fenêtre
fermée par l'utilisateur ne lève aucune exception : elle rend dans un contexte
OpenGL détruit, et VTK réclame alors un shader qu'il ne peut plus compiler —
d'où le flot d'`ERR| Could not create shader object` et
`attempt to add attribute without a program`.

Il faut donc demander à VTK si sa fenêtre existe encore, et de plusieurs
façons, aucune n'étant fiable seule d'une plateforme à l'autre : un
observateur sur `ExitEvent`, l'indicateur `_closed` du plotter, la présence de
`render_window`, et `GetDone()` sur l'interacteur. Le premier signal reçu
arrête le rafraîchissement et referme proprement.

### Cliquer et déplacer

Deux gestes que la capture d'écran interdisait :

* **cliquer un repère** — désigner l'encoche qu'on veut emprunter, décrit plus
  haut. Un clic à plus de 120 mm de tout repère ne désigne **rien** : prendre
  le plus proche quoi qu'il arrive ferait basculer un choix à l'autre bout de
  la maquette sur un clic de rotation manqué ;
* **déplacer une poignée** — l'édition manuelle ci-dessous.

### Édition manuelle du tracé (BETA)

La case *Édition manuelle (BETA)* pose des poignées déplaçables sur le tracé du
meilleur agent. Glisser l'une d'elles **impose** le point : les agents l'y
replacent à chaque itération, puis le retirent de ceux qu'ils peuvent bouger.
L'agent optimise donc autour de la décision de l'intégrateur au lieu de la
défaire au tour suivant.

Rien de neuf sous le capot : c'est la mécanique des encoches de peigne, ouverte
à l'utilisateur. Un point posé à la main et une entrée de peigne sont la même
contrainte, et suivent le même chemin — `snap_mandatory_points`, puis le gel de
l'indice.

Quelques décisions qui méritent d'être dites :

* **au plus quatorze poignées**, échantillonnées le long du tracé. Une par
  point serait illisible sur un faisceau de cinquante points, et surtout
  impossible à saisir : deux poignées voisines se recouvriraient ;
* **les extrémités n'en reçoivent pas.** A et B appartiennent aux équipements ;
* **les poignées suivent le tracé** à chaque rafraîchissement, sinon elles
  désigneraient un point que le câble a quitté ;
* **décocher la case ne libère pas les points.** Ils ont été posés
  délibérément, et les perdre au premier décochage ferait tout recommencer.
  *Libérer les points imposés* est une action à part — et elle existe : une
  contrainte qu'on ne peut plus retirer est un piège.

Ce qui justifie le BETA : le point imposé est respecté, mais rien ne vérifie
qu'il est *tenable*. Placé dans la structure, il y restera — les agents
l'honoreront et le rapport de conformité signalera le clash. C'est un outil
d'expert, pas un garde-fou.

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

Ces fixations sont **dessinées dans la vue 3D avec leur propre géométrie** :
le STL du modèle, recalé par la matrice que l'ICP a trouvée — ce que faisait
déjà l'ancienne application. Un repère symbolique ne dirait rien de
l'encombrement réel d'un peigne, qui est précisément ce que l'intégrateur doit
juger à l'œil. Le détecteur rend le chemin du fichier et la matrice de
recalage ; la mise en forme les laissait tomber, et il ne restait qu'un point à
dessiner. S'y ajoutent une bille verte à l'entrée de chaque encoche, une rouge
à la sortie, et le segment jaune que le câble doit emprunter. Un modèle
introuvable ou illisible retombe sur une bille grise : mieux vaut un repère
approximatif qu'une fixation disparue de la vue. Rien ne les dessinait
auparavant — le viewer savait masquer les acteurs `clamp_`, mais aucun n'était
jamais créé, et l'utilisateur n'avait aucun moyen de vérifier ce que le scan
avait reconnu.

#### Le couloir de cheminement

Le détecteur balaie **toute** la maquette : il reconnaît aussi bien les peignes
qui jalonnent le trajet que ceux montés à l'autre bout de l'appareil. Les
emprunter tous obligeait le câble à aller les chercher — le trajet ne
s'arrêtait plus là où il devait.

Un peigne n'est donc retenu que s'il est dans le **couloir de cheminement** :
l'ellipsoïde de foyers A et B, où le détour qu'il impose — `|PA| + |PB|` — ne
dépasse pas 1,25 fois la distance directe `|AB|` (`DEFAULT_ZONE_FACTOR`). Le
critère est relatif à la longueur du trajet, pas une distance en millimètres :
c'est la même chose qui compte sur un cheminement de 50 cm et sur un de 5 m.

Mesuré sur deux cadres alignés et un peigne reconnu dans la soute arrière
(détour ×3,91) : trajet de 7 530 mm sans filtre, 1 841 mm avec.

**Le même couloir retient les trajectoires.** Rien n'empêchait un point d'aller
au-delà de l'arrivée : `detour_penalty` juge la longueur **totale** et
répartit sa sanction uniformément, si bien qu'un point parti trois mètres trop
loin n'y contribue guère plus que ses voisins restés en place — il ne reçoit
donc aucun signal qui lui dise de revenir. `zone_penalty` sanctionne **point
par point**, sur la même ellipse : nulle partout dans le couloir, croissante
au-delà, saturée pour ne pas écraser les autres règles. A et B sont sur
l'ellipse de rapport 1 : ils ne sont jamais sanctionnés, quel que soit le
facteur retenu. Une seule notion de zone sert aux deux usages — deux
finiraient par diverger.

#### Choisir soi-même l'encoche

Une fois le scan terminé et les passages dessinés, l'application ouvre une
fenêtre : **par quelle encoche le faisceau doit-il passer ?** Une liste
déroulante par peigne, l'encoche que le calcul retiendrait déjà sélectionnée,
et « ne pas emprunter ce peigne » en tête de liste.

Une liste déroulante plutôt qu'une case par encoche : un peigne peut en porter
treize, et treize cases à cocher dont une seule peut être retenue est un
formulaire qui ment sur ce qu'il autorise. La liste dit la règle par sa forme
même — un choix, et un seul.

L'application propose, l'intégrateur tranche : c'est lui qui sait quelle
encoche est libre, laquelle est réservée à un autre faisceau, laquelle est
atteignable à l'outil, et rien de cela n'est dans le DMU. Chaque changement se
répercute **aussitôt sur la vue 3D** : l'encoche désignée s'allume avant même
de valider, plutôt que d'arbitrer sur des coordonnées. La fenêtre arrive après
l'affichage, jamais avant.

**Le choix se fait aussi directement en 3D.** La vue s'ouvre d'elle-même au
lancement du cheminement — poser la question devant une fenêtre fermée
reviendrait à demander d'arbitrer à l'aveugle — et chaque encoche y est
cliquable. Cliquer une encoche la retient pour son peigne ; recliquer celle
qui l'est déjà écarte le peigne. C'est le geste minimal : un peigne n'accepte
qu'une encoche, donc désigner, c'est choisir, et le seul autre état possible
est « aucune ». Le clic et la liste déroulante décrivent le même choix et
restent synchronisés — il ne doit pas en exister deux versions.

Tout ignorer, fermer la fenêtre et écarter chaque peigne un à un reviennent au
même : aucun passage imposé. La réponse se retrouve sur l'interrupteur
*Emprunter les fixations existantes* de la page, et y reste mémorisée ; le
réglage de la page décide en retour de ce que la fenêtre propose à l'ouverture.
Sans peigne dans le couloir, aucune question n'est posée.

Les billes d'entrée (verte) et de sortie (rouge) reprennent la taille de
l'ancienne application — la moitié du rayon du toron, assez petites pour ne pas
masquer l'encoche qu'elles repèrent — et le **segment vert** qui les
joint restent affichés pendant tout le cheminement : l'arrêt du calcul n'efface
que les tracés des agents et les crabes posés. Ce sont les repères de ce qui
est imposé, ils doivent rester lisibles quand la trajectoire bouge.

#### Une encoche par peigne, pas toutes

Un peigne porte plusieurs encoches **côte à côte**, une par faisceau : les
autres appartiennent aux voisins. Un faisceau en emprunte donc **une**. Le
trajet enfilait au contraire toutes les encoches détectées, ordonnées par la
projection de leur centre sur A→B — et comme les encoches d'un même peigne se
projettent au même endroit, elles se suivaient dans la liste. Le câble
traversait une encoche, ressortait, et repartait aussitôt dans celle d'à côté
au lieu de rejoindre le peigne suivant.

Deux libertés restent à exploiter, et elles ne se décident pas séparément :
**quelle** encoche prendre sur chaque peigne, et **dans quel sens** la
traverser — `p_in` et `p_out` sont interchangeables, seule leur solidarité
compte. `core/passage_route.py` tranche les deux ensemble, par programmation
dynamique sur les peignes ordonnés le long du trajet : le résultat est le
trajet le plus court parmi **tous** les choix d'encoche et de sens, vérifié
contre une énumération exhaustive. Un choix local — « l'encoche la plus
proche », « le côté rencontré en premier » — rejouerait le défaut sous une
autre forme : c'est justement en raisonnant encoche par encoche qu'on finit
par faire la navette sur un même peigne.

Mesuré sur deux peignes de six encoches qui se font face : dix allers-retours
sur un même peigne avant, aucun après ; trajet de 2 551 mm ramené à 1 840 mm.

Un peigne est situé dans le trajet par **son centre**, pas par celui d'une de
ses encoches : sa place ne doit pas dépendre de l'encoche qu'on lui aura
choisie, puisque ce choix vient après.

L'encoche retenue est visible partout : marquée d'une flèche dans la liste du
bandeau, seule à garder ses couleurs dans la vue 3D — les autres passent en
gris fin plutôt que d'être effacées, car elles existent — et nommée en console
avec son numéro et son sens de traversée. Le calcul ne fait que **proposer** :
la fenêtre décrite plus bas laisse l'utilisateur imposer une autre encoche, ou
écarter le peigne.

#### Un couple ne se disloque pas

**Répondre oui contraint deux choses.** Le chemin de départ est découpé en
tronçons par les couples entrée/sortie retenus ; la traversée de l'encoche
elle-même est une ligne droite de **deux points**, y lancer une recherche de
chemin ferait contourner le peigne au lieu de passer dedans. Et surtout, **les
agents y sont maintenus** : à chaque itération les couples sont replacés sur
la trajectoire, puis retirés des points que l'agent peut déplacer.

L'épinglage se fait **par couple**, jamais point par point. Un couple n'est
pas deux contraintes indépendantes, c'est une traversée : épingler chaque
point sur le point du câble le plus proche laisse le câble entrer dans une
encoche et ressortir par une autre — et sur un peigne à treize encoches
voisines, c'est le cas général, tous les candidats étant à quelques
centimètres. Les deux points d'un couple vont donc sur deux points
**consécutifs** du câble, et les couples se succèdent dans l'ordre du trajet.
Faute de place, un passage reste non épinglé : un couple coupé en deux serait
pire que pas de contrainte du tout.

Cette dernière contrainte n'est pas un excès de prudence. L'agent dispose déjà
d'une attraction par récompense vers les fixations existantes ; mesurée sur une
vraie boucle, elle ne suffit pas. Après deux cents itérations, le câble s'écarte
de **220 à 350 mm** des encoches — une encoche de peigne ne se négocie pas à
cette distance. Avec l'épinglage, l'écart mesuré est de **0,0 mm** sur tous les
passages, pour tous les agents.

Répondre non annule tout : ni découpage du trajet, ni épinglage, ni même
l'attraction par récompense — la liste des fixations n'est pas transmise aux
agents. Un refus doit être un vrai refus.

#### Ce que le scan raconte en console

« Aucune fixation trouvée » et « le scan n'a jamais eu lieu » se ressemblent
beaucoup vus de l'interface, et pas du tout une fois qu'il faut comprendre
pourquoi. Le scan journalise donc chacune de ses décisions : le dossier de
modèles retenu, le fichier de maquette réellement analysé, la liste nominative
des modèles cherchés, puis fixation par fixation le score de recalage, la
position et chaque passage `p_in`/`p_out`.

```
======================================================================
SCAN DES FIXATIONS EXISTANTES
======================================================================
📁 Dossier des modèles : /home/…/clamps
📐 Maquette réexportée pour le détecteur : /home/…/fusion/temp_for_detection.stl
🌍 Maquette analysée : /home/…/fusion/temp_for_detection.stl
🔎 2 modèle(s) STL à rechercher :
   • XA453420_peigne.stl
   • clip.STL
```

**Le scan analyse la maquette chargée, pas un fichier.** L'étape d'extraction
enregistre la fusion en `.vtk` ; Open3D ne lit pas ce format et rend un
maillage **vide sans lever d'erreur**. Le détecteur comparait donc les modèles
à un environnement inexistant et ne reconnaissait évidemment rien — en
apparence, un scan qui tourne et ne trouve jamais rien. La maquette déjà en
mémoire est désormais réexportée en STL avant le scan : ce qui est analysé est
exactement ce que parcourent les agents, quel que soit le format dans lequel la
fusion a été rangée. À défaut de maillage en mémoire, un fichier d'un format
non lisible est converti plutôt qu'envoyé tel quel.

Le détecteur repose sur Open3D. Sans lui, le scan ne plante pas : il dit
pourquoi il n'a pas eu lieu, et le cheminement continue sans fixations
préexistantes. Cinq causes d'abandon sont distinguées et nommées : dossier non
renseigné, dossier sans aucun STL, maquette illisible, Open3D absent, scan en
erreur — cette dernière recopiant la trace d'exécution en console.

### Les crabes posés par les agents

Les crabes ne sont pas ajoutés après coup : `compute_crabes` tourne **à chaque
itération**, à l'intérieur de la boucle d'apprentissage. Leur position pèse sur
la récompense (`R_crabe`, `R_fixation`) et alimente le rapport de conformité,
si bien que la trajectoire est déplacée pour aller là où l'on peut réellement
fixer.

Ils sont dessinés au fil du calcul **avec leur géométrie réelle** — le modèle
STL du crabe, pas un repère symbolique — et **dans le repère où leur absence de
collision a été vérifiée** : même matrice de rotation `[x_axis, y_axis, normal]`,
même origine sur la structure que `is_crabe_clash_free`. Dessiner ailleurs que
là où le test a eu lieu donnerait une vue rassurante et fausse ; un test croisé
vérifie que la géométrie affichée reste bien du bon côté de la surface.

Le modèle est normalisé une fois pour toutes au chargement : sa face de plus
grande aire devient le plan de contact, ramené en `z = 0` et centré. C'est ce
qui permet de le poser sur n'importe quelle surface sans se soucier de
l'orientation d'origine du fichier.

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
  passage_route.py          quelle encoche emprunter par peigne, dans quel sens
                            et quels peignes sont dans le couloir
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
tests/                      460 tests hors interface, 150 tests d'interface
```

`core/geometry_metrics.py`, `core/routing_rules.py`, `core/reward_terms.py` et
`core/orchestrator.py` ne dépendent que de numpy : ils sont testables sans
PyTorch, sans maillage et sans écran.

### L'ancienne application

`old_code/` contient l'application d'origine, fournie comme référence :
`controller.py`, `fixation.py` et `fixation_detection.py`. Ce dernier est
**identique** au détecteur du dépôt — ce qui confirme que le scan ne trouvait
rien à cause du format de maquette qu'on lui donnait, pas à cause de lui.

Deux points y ont été repris. La règle « une encoche par peigne, dans le sens
du trajet » : elle y était déjà, avec un critère plus simple — l'encoche dont
le centre est le plus proche du segment A–B, et un produit scalaire pour le
sens. La programmation dynamique décrite plus haut la généralise en tenant
compte de l'enchaînement des peignes, et se vérifie contre une énumération
exhaustive. Et la taille des billes, `tube_radius * 0.5`, pour que le repère
reste celui qu'on connaît.

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
