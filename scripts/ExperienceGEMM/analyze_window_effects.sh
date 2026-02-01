#!/bin/bash

# Analyse de l'effet de fenêtre (Window Effect Analysis)
# Objectif: Vérifier la stabilité des mesures entre 500 et 5000 itérations en 16 threads
# pour s'assurer que les résultats ne sont pas du bruit d'initialisation.

OUTPUT_DIR="output/ExperienceGEMM/window"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Expérience C: Analyse de l'effet de fenêtre (Window Effect)"
echo "=========================================="
echo ""

SIZE=128
THREADS=16
export MATMUL_IMPL=blas
export OPENBLAS_NUM_THREADS=$THREADS

# Fonction pour extraire valeur et CV
# Utilise LC_ALL=C pour éviter les problèmes de formatage de nombre (virgules, etc.)
perf_value() {
    # Grep robuste: cherche le mot clé, prend la 1ère ligne, extrait la 1ère colonne (valeur)
    # cut -d, -f1 gère le format CSV de perf -x,
    echo "$1" | grep -i "$2" | head -n 1 | cut -d, -f1
}

perf_cv() {
    # Cherche le mot clé, et essaie d'extraire le pourcentage (souvent colonne 4)
    # Mais le format est variable. On cherche le champ qui contient "%"
    echo "$1" | grep -i "$2" | head -n 1 | awk -F, '{for(i=1;i<=NF;i++) if($i~/%/) {gsub(/%/,"",$i); print $i; exit}}'
}

run_window_test() {
    REPS=$1
    echo "--- Test avec $REPS itérations ---"
    
    # Exécuter perf stat 5 fois pour obtenir moyenne et variance
    # Utilisation de taskset -c 0-7 pour cohérence avec les expériences précédentes
    # 2>&1 est CRITIQUE car perf stat écrit sur stderr
    OUTPUT=$(LC_ALL=C taskset -c 0-7 perf stat -r 5 -x, -e instructions,cycles,context-switches \
        ./build/test_benchmark_large $SIZE $SIZE $SIZE $REPS 2>&1)
        
    # Debug: En cas de problème, décommenter la ligne suivante
    # echo "$OUTPUT"
    
    # Note: perf -x, output format:
    # value,unit,event,variance,runtime,metric...
    # Field 1: value
    # Field 3: event name
    # Field 4: variance (if -r is used)
    
    INSTR=$(perf_value "$OUTPUT" "instructions")
    CYCLES=$(perf_value "$OUTPUT" "cycles")
    CS=$(perf_value "$OUTPUT" "context-switches")
    
    INSTR_CV=$(perf_cv "$OUTPUT" "instructions")
    CS_CV=$(perf_cv "$OUTPUT" "context-switches")
    
    # Check si les valeurs ont été extraites
    if [ -z "$INSTR" ]; then
        echo "ERREUR: Impossible d'extraire les données perf."
        echo "Sortie brute de perf stat:"
        echo "$OUTPUT"
        INSTR=0
        CYCLES=1
        CS=0
    fi
    
    # Calculs normalisés (par itération)
    if [ "$INSTR" != "0" ]; then
        INSTR_PER_REP=$(echo "$INSTR / $REPS" | bc)
        IPC=$(echo "scale=2; $INSTR / $CYCLES" | bc -l)
        CS_PER_REP=$(echo "scale=4; $CS / $REPS" | bc -l)
    else
        INSTR_PER_REP="N/A"
        IPC="N/A"
    fi

    echo "  Total Instructions: $INSTR"
    echo "  Instructions/Rep:   $INSTR_PER_REP"
    echo "  IPC:                $IPC"
    echo "  Total CS:           $CS"
    echo "  CS/Rep:             $CS_PER_REP"
    echo "  Variabilité (CV):   Instr=${INSTR_CV}%, CS=${CS_CV}%"
    echo ""
    
    # Sauvegarde pour comparaison
    echo "$REPS,$INSTR,$INSTR_PER_REP,$IPC,$CS,$CS_PER_REP,$INSTR_CV,$CS_CV" >> "${OUTPUT_DIR}/results.csv"
}

# Initialiser le fichier CSV
echo "Reps,Total_Instr,Instr_Per_Rep,IPC,Total_CS,CS_Per_Rep,Instr_CV,CS_CV" > "${OUTPUT_DIR}/results.csv"

# 1. Fenêtre courte (500 itérations)
run_window_test 500

# 2. Fenêtre longue / Standard (2000 itérations)
run_window_test 2000

echo "=========================================="
echo "Comparaison et Conclusion"
echo "=========================================="

# Lire les résultats pour affichage final
# On utilise awk pour formatter un joli tableau
echo -e "\nReps\tInstr/Rep\tIPC\tCS/Rep\tInstr_CV"
awk -F, 'NR>1 {printf "%d\t%.0f\t\t%.2f\t%.4f\t%.2f%%\n", $1, $3, $4, $6, $7}' "${OUTPUT_DIR}/results.csv"

echo ""
echo "Critères de stabilité:"
echo "1. Instr/Rep doit être similaire (indique que le code exécuté est le même)"
echo "2. IPC doit être stable (indique que le comportement dynamique est constant)"
echo "3. CV faible (< 5%) indique des mesures fiables"

# Conclusion automatique basée sur la variance
CV_500=$(awk -F, 'NR==2 {print $7}' "${OUTPUT_DIR}/results.csv")
CV_2000=$(awk -F, 'NR==3 {print $7}' "${OUTPUT_DIR}/results.csv")

if [ -z "$CV_500" ] || [ -z "$CV_2000" ]; then
    echo "⚠️ Conclusion: CV indisponible, impossible d'évaluer la stabilité."
    exit 0
fi

if (( $(echo "$CV_500 > 5" | bc -l) )); then
    if (( $(echo "$CV_2000 > 5" | bc -l) )); then
        if (( $(echo "$CV_2000 < $CV_500" | bc -l) )); then
            echo "⚠️ Conclusion: Instabilité détectée (variance élevée). Converge mais reste instable."
        else
            echo "⚠️ Conclusion: Instabilité détectée (variance élevée)."
        fi
    else
        echo "⚠️ Conclusion: Instabilité détectée à court terme, mais stabilisation à fenêtre standard (2000)."
    fi
else
    if (( $(echo "$CV_2000 > 5" | bc -l) )); then
        echo "⚠️ Conclusion: Instabilité détectée (variance élevée) à fenêtre standard."
    else
        echo "✓ Conclusion: Mesures stables (variance faible)."
    fi
fi
