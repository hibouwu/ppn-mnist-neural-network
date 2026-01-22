# Guide des scripts de test

## Préparer l'environnement expérimental

```bash
# 1. Préparer l'environnement (sudo requis)
# ------------------------------------------------------------------
echo "Configuration de l'environnement..."
# Placer le gouverneur CPU en mode performance
sudo cpupower frequency-set -g performance
# Verrouiller la fréquence à 4.0 GHz (4000MHz)
sudo cpupower frequency-set -u 4000MHz -d 4000MHz

# Vérifier la fréquence actuelle
echo "Fréquence CPU actuelle :"
cpupower frequency-info | grep "current CPU frequency"
```

## Tester la multiplication matricielle avec différents threads et tailles

```bash
# 2. Lancer le script de test
# ------------------------------------------------------------------
echo "Début des tests..."
./find_optimal_threads.sh
# ou
# Lier tout le script aux 8 premiers cœurs (0-7)
# Tous les sous-processus héritent de ce binding
sudo taskset -c 0-7 bash scripts/find_optimal_threads.sh
```

Les résultats sont dans [output/outputresult/thread_scaling.csv](output/outputresult/thread_scaling.csv).
Conclusions :

1. **Instabilité des petites matrices (Small Matrix Instability)** :
   - **Observation** : lors des premiers tests, la boucle externe en Bash introduisait de fortes variations sur les petites matrices (64x64) même en mono-thread.
   - **Optimisation** : la boucle a été déplacée **à l'intérieur du C++ (Internal Loop)**, supprimant le bruit du lancement de processus et de l'ordonnanceur.
   - **Résultat** : l'écart type tombe au **niveau microseconde (~2us)**, prouvant qu'avec une méthodologie correcte, les petites tailles sont très stables.
   - **Conclusion** : malgré la stabilité, pour les matrices < 256x256 il est recommandé d'utiliser `OMP_NUM_THREADS=1`, car le overhead Fork/Join annule ou dépasse le gain du multithreading.

   > **Étude de cas : Active Spin vs Passive Wait**
   > Même sans bruit de mesure, on observe :
   > - **8 threads (Active Spin)** : quand threads <= cœurs physiques, OpenMP busy-wait et devient très sensible au bruit système.
   > - **16 threads (Passive Wait)** : quand threads > cœurs physiques, OpenMP passe en attente passive (yield) et la stabilité moyenne s'améliore.

2. **Méthodologie statistique** :
   - **Moyenne interne** : chaque point exécute 50-100 itérations dans C++ et prend moyenne et écart type haute précision.
   - **Warm-up** : 5 runs à vide avant mesure pour chauffer i-cache et d-cache.
   - **Moyenne tronquée** : le script de tracé agrège plusieurs runs et coupe les 5% extrêmes pour robustifier.

3. **Affinité cœur et sursouscription** :
   - **Contexte** : les expériences utilisent `taskset -c 0-7`, limitant strictement à 8 cœurs physiques.
   - **Constat clé** : le meilleur nombre de threads est **fixé à 8**.
   - **Pénalité d'oversubscription** : avec 16 threads, l'OS fait du context switch sur 8 cœurs.
     - Pour **BLAS** (pipelines saturés), chaque switch est une perte pure : **~40% plus lent** (0.05s → 0.08s sur 2048x2048).

4. **BLAS vs OpenMP maison** :
   - **BLAS** : utilise fortement le CPU (AVX/FMA), très sensible à la limite de cœurs, s'effondre en oversubscription.
   - **OpenMP maison** :
     - Notre `dgemm_omp` parallélise mais la pipeline mono-thread est moins efficace que BLAS (bulles/attentes).
     - Donc à 16 threads sur 8 cœurs, le coût de switch peut parfois masquer des bulles et ne pas dégrader autant que BLAS, voire être légèrement plus rapide sur les grandes tailles.

5. **Recommandations grandes matrices** :
   - Pour >= 512x512, le multithreading apporte un gain net.
   - Dans notre environnement limité, **8 threads** est recommandé.

6. **Recommandations résumées** :
   - **OpenMP** :
     - **Petites matrices (< 128x128)** : imposer `OMP_NUM_THREADS=1`.
     - **Moyennes (256x256 à 512x512)** : 4 threads, bon compromis gain/stabilité.
     - **Grandes (>= 1024x1024)** : égal aux cœurs physiques (8 ici).
     - **Pour la suite, nous testerons par défaut 4 et 8 threads.**
   - **BLAS** : utilisation CPU extrême, très sensible à l'oversubscription, donc threads = cœurs (8 ici).
     - Sur petites matrices, l'impact du nombre de threads est faible.
     - Sur grandes matrices, choisir le nombre de cœurs physiques (8 ici).
   - **Pour les matrices 28x28 (ou 784x1) du projet** :
     - Préconiser BLAS mono-thread.

```bash
# Générer les graphiques
python3 scripts/plot_scaling.py
```

Résultats : [output/outputresult/scaling_plot.png](output/outputresult/scaling_plot.png) et [output/outputresult/scaling_speedup_plot.png](output/outputresult/scaling_speedup_plot.png)

![temps plot](output/outputresult/scaling_plot.png)
![scaling speedup plot](output/outputresult/scaling_speedup_plot.png)

### Lancer le test d'affinité

```bash
# Test d'affinité (matrice 28x28)
# ------------------------------------------------------------------
# Parcourt automatiquement différents nombres de threads et stratégies (default, close, spread)
echo "Running Affinity Benchmark..."
./scripts/benchmark_affinity.sh

# Résultats dans output/affinity_comparison.csv
cat output/affinity_comparison.csv
```

**Test d'affinité (cas 28x28)** :
    -   **Contexte** : matrices 28x28, on teste `OMP_PROC_BIND` & `OMP_PLACES`.
    -   **Résultats** :
        -   **OpenMP** : `close` est meilleur à 4 threads (2.53us), devant `spread` (4.43us) et `default` (3.92us). Limiter la communication inter-cœur est vital sur si petite charge.
        -   **BLAS** : temps constant ~0.81us, preuve d'un micro-noyau mono-thread ultra optimisé insensible aux réglages externes.
    -   **Conclusion** : pour 28x28, **BLAS est imbattable** (0.8us vs 2.5us pour le meilleur OMP). Si OpenMP est imposé : `4 threads + OMP_PROC_BIND=close`.

## Tester l'impact des niveaux d'optimisation sur la multiplication

```bash
# 2. Lancer le test des niveaux d'optimisation
# ------------------------------------------------------------------
echo "Début des tests d'optimisation..."
./scripts/benchmark_large.sh
# ou
# Lier le script aux 8 premiers cœurs (0-7)
sudo taskset -c 0-7 bash scripts/benchmark_large.sh
```

Résultats : [output/outputresult/impl_comparison.csv](output/outputresult/impl_comparison.csv)

```bash
# Générer les graphiques de comparaison et d'accélération
python3 scripts/plot_comparison.py
```

Résultats : [output/outputresult/comparison_grid_plot.png](output/outputresult/comparison_grid_plot.png) et [output/outputresult/comparison_speedup_grid.png](output/outputresult/comparison_speedup_grid.png)

![comparison grid plot](output/outputresult/comparison_grid_plot.png)
![comparison speedup grid](output/outputresult/comparison_speedup_grid.png)

Conclusions :

1. **Hiérarchie de performance** :
   - **Naïf (`ijk`)** : très lent, explosion $O(N^3)$ ; plusieurs secondes à 2048x2048.
   - **Réordonné (`ikj`)** : simple changement d'ordre tire parti de la localité cache, gros gain.
   - **OpenMP maison** : le parallélisme apporte un gain d'ordre de grandeur.
   - **BLAS (OpenBLAS)** : SIMD + optimisations asm, plus de **1200x plus rapide** que l'implémentation naïve.

2. **Stratégie de visualisation (linéaire + unités adaptatives)** :
   - Échelle **linéaire**, pas logarithmique, pour montrer le gouffre (`ijk` très haut, les autres au ras du sol).
   - **Unités adaptatives** :
     - Petites matrices (64x64) en **microsecondes (us)**.
     - Grandes matrices (2048x2048) en **secondes (s)**.
     - On garde ainsi lisibilité micro + macro.

3. **Accélération (Speedup)** :
   - Les “marches” d'accélération sont claires ; BLAS atteint **~1200x** sur la plus grande taille, preuve que l'optimisation algorithmique l'emporte sur le seul hardware.

4. **Recommandations** :
   - **Optimisation algorithmique** : réordonnancement et parallélisation sont clés.
   - **Bibliothèques optimisées** : pour l'usage réel, préférer OpenBLAS / Intel MKL, etc.
   - **Dans ce projet (matrices 28x28), le meilleur speedup est 15.3x avec BLAS 8 threads**, mais vu la faible stabilité et le faible gain du multithreading sur petites tailles, nous retenons BLAS mono-thread comme option par défaut.

## Profiling des multiplications en cours d'entraînement

Pour analyser les goulots d'étranglement, nous profilons l'implémentation **Naïve** et **BLAS** sur toute la chaîne.

### 1. Activer le profiling et compiler

Assurez-vous que `-pg` est activé dans `CMakeLists.txt`.

```bash
# 1. Régler le CPU à 4.0GHz
# ------------------------------------------------------------------
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -u 4000MHz -d 4000MHz

# 2. Recompiler (Release + profiling)
# ------------------------------------------------------------------
# Vérifier -pg actif et ENABLE_PROFILE_MATMUL=OFF pour éviter le spam
cd build
make clean
cmake .. -DENABLE_PROFILE_MATMUL=OFF -DCMAKE_BUILD_TYPE=Release
make -j8 ppn_train
cd ..
```

### 2. Lancer le profiling (implémentations séparées)

On peut automatiser la comparaison des 5 implémentations (blas, ijk, ikj, blocked, omp) grâce au script `scripts/benchmark.sh`.

```bash
# 1. Configurer l'environnement (Fréquence stable + Threads limités)
# ------------------------------------------------------------------
# Verrouiller à 4GHz (adapté à votre CPU)
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -u 4000MHz -d 4000MHz

# Limiter à 8 threads (ou 4) pour éviter l'overhead
export OMP_NUM_THREADS=8

# 2. Lancer le benchmark (avec taskset)
# ------------------------------------------------------------------
# Le script va compiler les résultats dans un tableau comparatif
taskset -c 0-7 ./scripts/benchmark.sh


# Impl       | Wall-Time       | CPU-Time        | Top-Gprof-Function            
# --------------------------------------------------------------------------------------
# blas       | 0:18.37         | 113.99          | std::vector<double,           
# ijk        | 1:19.00         | 78.25           | (anonymous                    
# ikj        | 0:49.06         | 48.45           | (anonymous                    
# blocked    | 0:56.13         | 55.50           | (anonymous                    
# omp        | 0:21.86         | 152.45          |                               

# detailed logs: build/train_*.log
# profile reports: build/gprof_*.txt



# 3. Restaurer l'environnement
# ------------------------------------------------------------------
sudo cpupower frequency-set -g powersave
sudo cpupower frequency-set -d 421MHz -u 5386MHz  # Adaptez aux limites de votre CPU
```

### 3. Analyse détaillée des temps (Décomposition)

L'écart entre le temps Wall-clock (`/usr/bin/time`) et le temps CPU (`gprof`) permet d'isoler le coût de la multiplication matricielle par rapport au reste du programme (I/O, allocations, autres calculs).

| Implémentation | Wall-Clock ($T_{total}$) | Matmul (Gprof) | "Reste" ($T_{total} - T_{matmul}$) | Part Matmul |
| :--- | :--- | :--- | :--- | :--- |
| **Naïf (ijk)** | **79.00s** | 61.69s | **17.31s** | 78\% |
| **Bloqué** | **56.13s** | 38.15s | **17.98s** | 68\% |
| **Réordonné (ikj)** | **49.06s** | 31.72s | **17.34s** | 65\% |
| **OpenMP** | **21.86s** | $\sim$ 4.4s (est.) | $\sim$ 17.5s | **~20\%** |
| **BLAS** | **18.37s** | $\sim$ 1s (est.) | $\sim$ 17.4s | **~5\%** |

**Observations clés :**

1. **Constante du "Reste"** : Le temps hors-matmul est remarquablement stable ($\sim$ 17.5s) sur les versions séquentielles. Cela inclut le chargement des données (I/O) et la gestion mémoire.
2. **Impact BLAS** : En supposant ce coût fixe, l'opération Matmul ne prend plus que **~1 seconde** avec OpenBLAS, contre **>60 secondes** avec la version naïve.
3. **Nouveau Goulot** : Avec BLAS, 95\% du temps est désormais passé dans les tâches annexes (ex: I/O, Allocations). L'optimisation du calcul est terminée ; la suite nécessiterait d'optimiser les E/S (Loi d'Amdahl).

## Autres scripts : générer un diagramme PlantUML

```bash
python3 encode_plantuml.py output/thread_scaling.csv
```

## Autre script

```bash
# 1. Verrouiller la fréquence CPU à 4.0GHz (sudo requis)
echo "Setting CPU frequency..."
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -u 4000MHz -d 4000MHz

# 2. Recompiler (Release + profiling)
sudo taskset -c 0-7 bash scripts/find_optimal_threads.sh

# 4. Générer le rapport gprof
echo "Generating gprof report..."
gprof build/ppn_train gmon.out > analysis_final.txt
echo "Report saved to analysis_final.txt"

# 5. Restaurer l'environnement CPU
echo "Restoring CPU environment..."
sudo cpupower frequency-set -g powersave
sudo cpupower frequency-set -d 421MHz -u 5386MHz

echo "All done!"
```

## Manuel de l'entraînement testé

### 1. Compilation

Assurez-vous d'être à la racine du projet :

```bash
mkdir -p build
cd build
cmake ..
make ppn_train
cd ..
```

### 2. Exemples d'utilisation

```bash
./scripts/exp_learning_rate.sh
python3 scripts/analyze_results_lr.py
python3 scripts/plot_lr_curves.py
```
