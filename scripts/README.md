# Guide des scripts de test

## Expériences GEMM (Multiplication Matricielle)

Pour les expériences sur la multiplication matricielle (thread scaling, optimisations, profiling), voir [scripts/ExperienceGEMM/README.md](ExperienceGEMM/README.md).

## Autres scripts : générer un diagramme PlantUML

```bash
python3 scripts/Utils/encode_plantuml.py output/thread_scaling.csv
```

## Autre script

```bash
# 1. Verrouiller la fréquence CPU à 4.0GHz (sudo requis)
echo "Setting CPU frequency..."
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -u 4000MHz -d 4000MHz

# 2. Recompiler (Release + profiling)
sudo taskset -c 0-7 bash scripts/ExperienceGEMM/find_optimal_threads.sh

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
./scripts/ExperienceHPO/exp_learning_rate.sh
python3 scripts/ExperienceHPO/analyze_results_lr.py
python3 scripts/ExperienceHPO/plot_lr_curves.py
```
