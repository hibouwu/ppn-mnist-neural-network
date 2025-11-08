# Spécification des besoins

## 1. Description des besoins du projet

### 1.1 Objectifs
- Développer un perceptron multicouche (MLP) en C++ capable de reconnaître les chiffres manuscrits MNIST.
- Construire un moteur de différentiation automatique en mode inverse pour générer automatiquement les gradients du modèle.
- Intégrer l’ensemble de la chaîne d’entraînement (propagation avant, rétropropagation, SGD, fonctions d’activation, initialisation) afin de livrer un solveur MNIST opérationnel à la fin du premier semestre.
- Préparer les bases nécessaires aux améliorations de performances et de précision prévues pour le second semestre.

### 1.2 Livrables du projet
- Une bibliothèque / un outil C++ fournissant une implémentation MLP pour MNIST, comprenant le moteur d’autodifférentiation, les algorithmes d’optimisation essentiels et le pipeline de lecture des données.
- Une documentation technique assortie d’exemples d’utilisation pour guider l’entraînement et l’évaluation du modèle.

### 1.3 Fonctionnalités du produit
- Propagation avant MLP configurable (nombre de couches, taille, fonctions d’activation ajustables).
- Moteur d’autodifférentiation en mode inverse permettant de générer le graphe de calcul et les gradients associés.
- Prise en charge des opérations vecteur/matrice (produit matriciel, sommes élément par élément, produits élément par élément, activations élémentaires) et calcul de leurs dérivées partielles.
- Boucle d’entraînement intégrant la mise à jour des poids/biais par SGD, une initialisation aléatoire contrôlable et le suivi des performances.
- Intégration du processus de chargement des données MNIST avec une interface d’inférence pour prédire les chiffres manuscrits.

### 1.4 Critères d’acceptation et de réception
- Atteindre une précision au moins égale au seuil défini sur le jeu de validation MNIST (exemple : ≥ 90 % à la fin du premier semestre).
- Vérifier l’exactitude des gradients via une comparaison par différences numériques sur des cas de test sélectionnés.
- Répondre aux exigences de performance sur la plateforme cible (temps d’entraînement raisonnable) et réussir les tests unitaires des opérations critiques.
- Fournir une documentation complète permettant d’exécuter le pipeline d’entraînement et d’inférence de bout en bout selon les instructions.

## 2. Contraintes

### 2.1 Contraintes de planning
- **08/11 Fin de la phase de conception**
  - Livrer l’architecture globale du système (découpage en modules, flux de données, dépendances) ainsi que la documentation des interfaces.
  - Valider les choix technologiques (C++17, OpenBLAS, cadre de tests, etc.) et les scripts de mise en place de l’environnement.
  - Produire un plan d’itération détaillé et une liste des risques, validés par la revue d’équipe.
- **16/11 Première version exécutable**
  - Finaliser la propagation avant du MLP de base, le moteur d’autodifférentiation en mode inverse et l’ossature SGD/fonction de perte, puis les intégrer.
  - Réussir les tests unitaires des opérations matricielles/vecteur et du calcul de gradients avec une erreur inférieure à 1e-5 par rapport aux différences numériques.
  - Réaliser un cycle complet avant + arrière + mise à jour des paramètres sur un mini-batch MNIST échantillonné, avec une perte décroissante et un guide d’exécution.
- **07/12 Deuxième version**
  - Intégrer la chaîne d’entraînement MNIST complète (chargement des données, entraînement mini-batch, enregistrement des métriques) avec une architecture de réseau configurable.
  - Atteindre ≥ 90 % de précision sur le jeu de validation ou fournir les raisons de l’écart et le plan d’amélioration.
  - Produire un premier jet de la documentation (guide d’utilisation, rapport de tests, problèmes connus) prêt pour démonstration et livraison.

### 2.3 Contraintes matérielles
- Ordinateur sous Linux avec VSCode comme environnement de développement.

### 2.4 Autres contraintes
- C++
- Bibliothèque OpenBLAS
- _À compléter : normes techniques, dispositions légales, exigences de sécurité, etc._

## 3. Mise en œuvre du projet

### 3.1 Planification
- **Premier semestre — Implémentation des fonctionnalités clés**
  1. Développer un MLP de base disposant de la propagation avant.
  2. Construire un moteur de différentiation automatique en mode inverse couvrant les opérations vecteur/matrice (somme, produit) ainsi que les fonctions d’activation élémentaires, y compris le calcul de leurs dérivées partielles.
  3. Intégrer l’autodifférentiation au MLP pour calculer les gradients.
  4. Mettre en place la boucle d’entraînement incluant SGD (gestion des poids, biais, fonctions d’activation).
  5. Intégrer le pipeline MNIST complet en s’appuyant sur le code de lecture des données fourni.

- **Second semestre — Optimisation des performances et de l’efficacité**
  - Améliorer la précision et la vitesse d’entraînement (architecture du modèle, régularisation, réglage des hyperparamètres, etc.).

### 3.2 Allocation des ressources
- Ressources humaines : 4 développeurs C++ et un encadrant pour l’appui technique.
- Ressources matérielles : environnement de développement compatible C++17+, bibliothèques standard usuelles, puissance de calcul suffisante pour entraîner MNIST (CPU multi-cœur, GPU optionnel).

# Projet « Autodiff + MLP + MNIST »

## 🎯 Objectifs globaux
1. Développer un MLP de base doté de la propagation avant.
2. Construire un moteur d’autodifférentiation en mode inverse prenant en charge les opérations matricielles/vecteur et les fonctions d’activation, avec calcul des dérivées.
3. Intégrer l’autodifférentiation au MLP pour réaliser le calcul des gradients.
4. Implémenter la boucle d’entraînement incluant SGD, avec gestion des poids, biais et fonctions d’activation.
5. Intégrer le pipeline d’entraînement et d’évaluation MNIST à l’aide du code fourni.

---

## 🧩 Phase 1 : Propagation avant du MLP de base

### 1. Structures de données et fondements mathématiques
- [ ] Implémenter les classes `Matrix` / `Vector` (addition, multiplication, transposition, diffusion).
- [ ] Ajouter l’initialisation aléatoire et les fonctions d’affichage.
- [ ] Rédiger des tests unitaires pour valider la justesse des opérations matricielles.

### 2. Module de fonctions d’activation
- [ ] Définir la classe abstraite `ActivationFunction` (interface : `forward`, `backward`).
- [ ] Implémenter `ReLU`, `Sigmoid`, `Tanh`.
- [ ] Écrire des tests pour vérifier la correction des sorties.

### 3. Module de couche linéaire
- [ ] Définir la classe `LinearLayer(in_dim, out_dim)`.
- [ ] Implémenter `forward(x) = x @ W + b`.
- [ ] Ajouter une initialisation aléatoire (distribution gaussienne ou uniforme).
- [ ] Tester la cohérence des dimensions en entrée et en sortie.

### 4. Assemblage du MLP de base
- [ ] Définir la classe `MLPNetwork` combinant `Linear + Activation`.
- [ ] Implémenter `addLayer()` et `forward(input)`.
- [ ] Vérifier manuellement le bon fonctionnement de la structure du réseau.

---

## ⚙️ Phase 2 : Moteur de différentiation automatique

### 5. Fondamentaux des nœuds et du graphe de calcul
- [ ] Implémenter la classe `Node` : `value`, `grad`, `parents`, `backward_fn`.
- [ ] Prendre en charge `backward()` avec un tri topologique automatique pour la rétropropagation du gradient.
- [ ] Tester les résultats de la rétropropagation sur l’addition / multiplication scalaires.

### 6. Prise en charge des opérations vecteur/matrice
- [ ] Implémenter `add`, `mul` (opérations élément par élément) et leurs règles de rétropropagation.
- [ ] Implémenter `matmul` (produit matriciel) et les règles inverses :
  - `dA += grad_output @ Bᵀ`
  - `dB += Aᵀ @ grad_output`
- [ ] Vérifier la correction du gradient du produit matriciel.

### 7. Autodifférentiation des fonctions d’activation
- [ ] Enregistrer `relu`, `sigmoid`, `tanh` dans le cadre d’autodifférentiation.
- [ ] Implémenter les règles inverses correspondantes.
- [ ] Validar les résultats par contrôle via gradient numérique.

### 8. Opérations d’agrégation
- [ ] Implémenter les opérations `sum`, `mean` et leur rétropropagation.
- [ ] Garantir la bonne diffusion (broadcast) des gradients.

---

## 🔁 Phase 3 : Intégration de l’autodifférentiation dans le MLP

### 9. Réécriture de la propagation avant du MLP pour la construction du graphe
- [ ] Remplacer `Matrix` par `TensorNode`.
- [ ] Baser les opérations de chaque couche sur les nœuds d’autodifférentiation.

### 10. Calcul du gradient du MLP
- [ ] Appeler `loss.backward()` pour calculer automatiquement les gradients.
- [ ] Extraire `.grad` à partir des nœuds de paramètres (W, b).

### 11. Validation des gradients
- [ ] Comparer les résultats de l’autodifférentiation par différences numériques.
- [ ] Tester la correction sur des fonctions simples (régression linéaire).

---

## ⚙️ Phase 4 : Mécanismes d’entraînement (SGD + initialisation + activation)

### 12. Module de fonction de perte
- [ ] Implémenter `MSELoss(pred, target)`.
- [ ] (Optionnel) Implémenter `CrossEntropyLoss`.
- [ ] Vérifier la justesse de la perte et de la rétropropagation.

### 13. Module d’optimisation
- [ ] Définir la classe abstraite `Optimizer`.
- [ ] Implémenter `SGDOptimizer` (avec `step()` et `zero_grad()`).
- [ ] Vérifier la tendance décroissante sur une fonction simple.

### 14. Stratégies d’initialisation
- [ ] Implémenter les méthodes d’initialisation `Xavier` et `Kaiming`.
- [ ] Choisir automatiquement la stratégie adaptée à la fonction d’activation.

### 15. Boucle d’entraînement complète
- [ ] Définir `train_step()` : propagation avant → perte → rétropropagation → mise à jour.
- [ ] Afficher la valeur de la perte à chaque itération.
- [ ] Confirmer la diminution continue de la perte durant l’entraînement.

---

## 📊 Phase 5 : Intégration du jeu de données MNIST

### 16. Chargement des données
- [ ] Utiliser les fonctions fournies pour charger les données MNIST.
- [ ] Normaliser les entrées sur [0,1] ou [-1,1].
- [ ] Implémenter un `DataLoader` (générateur de mini-batch).

### 17. Définition de l’architecture du réseau
- [ ] Construire la structure suivante :

---

784 → 128 → 64 → 10  
Activation : ReLU  
Perte : CrossEntropy

---

- [ ] Initialiser les poids et les biais.

### 18. Entraînement et évaluation
- [ ] Entraîner plusieurs époques sur MNIST.
- [ ] Enregistrer les courbes de perte et de précision.
- [ ] Évaluer la précision sur le jeu de test.

---

## 🧠 Phase 6 : Validation et documentation

### 19. Vérification de la justesse
- [ ] Contrôler l’absence de gradients nuls/explosifs.
- [ ] Confirmer la diminution de la perte au fil des époques.
- [ ] Évaluer la précision finale (objectif ≥ 88–92 %).

### 20. Expérimentations et analyses
- [ ] Tracer les courbes Loss/Accuracy.
- [ ] Comparer les effets des différentes initialisations et fonctions d’activation.

### 21. Documentation et livraison
- [ ] Rédiger la documentation de conception et les diagrammes UML.
- [ ] Mettre à jour le README avec les instructions d’exécution.
- [ ] Produire le rapport d’expérimentation et les graphiques des résultats.

---

