# HarnessOpt

Cheminement assisté de harnais électriques dans des cellules d'hélicoptère.

Le principe : partir d'un chemin géodésique sur la maquette numérique, puis
faire déplacer les points de ce chemin par une équipe d'agents d'apprentissage
par renforcement, jusqu'à obtenir un tracé qui respecte les règles
d'intégration (aucune interférence, distances tenues, rayons de cintrage
admissibles, fixations tous les 250 mm).

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
| **2. Règles** | Régler diamètre du toron, rayon de cintrage, distances mini/maxi, distances renforcées par famille, pas entre fixations. |
| **3. Cheminement** | Poser départ et arrivée, choisir une équipe d'agents et le réglage exploration/exploitation, lancer, suivre en direct. |
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
  reward_terms.py           traduction des règles en signal d'apprentissage
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
  charts.py  viewer3d.py    courbes de progression, vue 3D
  widgets/  pages/          composants et écrans
tests/                      133 tests hors interface, 30 tests d'interface
```

`core/geometry_metrics.py`, `core/routing_rules.py`, `core/reward_terms.py` et
`core/orchestrator.py` ne dépendent que de numpy : ils sont testables sans
PyTorch, sans maillage et sans écran.

### Fichiers hérités

`ui/main_window.py`, `ui/pages/extraction_view.py`, `ui/pages/agent_view.py`,
`controller/controller.py` et `core/controller/controller.py` sont l'ancienne
interface et son contrôleur. Ils ne sont plus appelés par `main.py`. Les deux
contrôleurs sont deux copies quasi identiques du même fichier ; ils importent
`core.catia_handler`, absent du dépôt, et ne s'importent donc pas en l'état.
Ils sont conservés pour référence — à supprimer quand la nouvelle interface
vous convient.

`core/sphere_generation.py`, `core/tools.py`, `core/visualize.py`,
`core/mesh_fusion.py`, `core/HS9019.py`, `core/smooth.py`,
`core/path_managment/fixation.py` et `core/path_managment/smooth.py`
appartiennent à une approche antérieure par graphe et ne sont importés par
aucun chemin actif.

---

## Tests

```bash
python -m pytest tests/ -q                    # règles, géométrie, agents
xvfb-run -a python -m pytest tests/test_ui.py # interface (Linux sans écran)
```

Les tests d'interface s'ignorent d'eux-mêmes si tkinter, customtkinter ou un
serveur graphique manquent.
