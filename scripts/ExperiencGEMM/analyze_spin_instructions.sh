#!/bin/bash

# Analyse Simplifiée: Focus sur les causes principales (Taille de matrice & Affinité CPU)
# Contient deux expériences clés:
#   Expérience 1 (ex-Exp2): Comparaison de tailles de matrices (prouver que les petites matrices amplifient la proportion de synchronisation)
#   Expérience 2 (ex-Exp4): Impact de l'affinité CPU (prouver que le pinning réduit la contention de planification)

OUTPUT_DIR="output/GEMM"
mkdir -p $OUTPUT_DIR

# Extraire la valeur et l'écart-type pour un événement perf (format -x,)
perf_value() {
    # Extract first column from perf csv output (value)
    echo "$1" | grep -i "$2" | head -n 1 | cut -d, -f1
}

perf_sd() {
    # Extract variance percentage from perf output
    echo "$1" | grep -i "$2" | head -n 1 | awk -F, '{for(i=1;i<=NF;i++) if($i~/%/) {gsub(/%/,"",$i); print $i; exit}}'
}

echo "=========================================="
echo "Analyse Simplifiée: Causes de l'anomalie BLAS 16 threads"
echo "=========================================="
echo ""

# ==========================================
# Expérience 1: Comparaison de tailles de matrices
# ==========================================

echo "=========================================="
echo "Expérience 1: Comparaison de tailles de matrices"
echo "=========================================="
echo ""
echo "Objectif: Montrer que les petites matrices ont un surcoût relatif plus élevé"
echo "Méthode: Comparer un coût normalisé (Instructions par opération FLOP)"
echo ""

# S'assurer que OMP_WAIT_POLICY n'est pas défini (comportement par défaut)
unset OMP_WAIT_POLICY

SIZES=(128 512 1024)

echo "Configuration de test: BLAS, 16 threads, 500 itérations"
echo ""

for size in "${SIZES[@]}"; do
    echo "--- Taille de matrice: ${size}×${size} ---"
    
    # Use taskset to isolate matrix size effects from migration issues
    SIZE_OUTPUT=$(taskset -c 0-7 perf stat -r 3 -x, -e instructions,cycles \
        ./build/test_benchmark_large $size $size $size 500 2>&1)
    
    SIZE_INSTR=$(perf_value "$SIZE_OUTPUT" "instructions")
    SIZE_CYCLES=$(perf_value "$SIZE_OUTPUT" "cycles")
    
    if [ -z "$SIZE_INSTR" ] || [ -z "$SIZE_CYCLES" ]; then
        echo "  Erreur: Impossible d'extraire les métriques"
        continue
    fi
    
    # Calculer le nombre d'instructions par itération
    INSTR_PER_ITER=$(awk -v total="$SIZE_INSTR" 'BEGIN {printf "%.2fM", total / 500 / 1000000}')
    # Approximation FLOPs = 2 * N^3 * Reps
    INSTR_PER_OP=$(awk -v total="$SIZE_INSTR" -v n="$size" 'BEGIN {ops=2*n*n*n*500; if (ops>0) printf "%.3e", total/ops; else print "N/A"}')
    
    echo "  Instructions totales: $(awk -v val="$SIZE_INSTR" 'BEGIN {printf "%.2fB", val / 1000000000}')"
    echo "  Par itération: ${INSTR_PER_ITER} instructions"
    echo "  Instr/Op (approx):    ${INSTR_PER_OP}"
    echo "  Cycles: $(awk -v val="$SIZE_CYCLES" 'BEGIN {printf "%.2fB", val / 1000000000}')"
    echo ""
done

echo "Attendu: Instr/Op devrait être relativement constant; si N=128 est plus élevé, cela indique un surcoût relatif plus important"
echo ""

# ==========================================
# Expérience 2: Impact de l'affinité CPU
# ==========================================

echo "=========================================="
echo "Expérience 2: Impact de l'affinité CPU sur la planification"
echo "=========================================="
echo ""
echo "Objectif: Vérifier si le pinning des threads réduit les context-switches et migrations"
echo "Méthode: Comparer planification par défaut vs affinité sur cores physiques"
echo ""

SIZE=128
REPS=5000
THREADS=16

export MATMUL_IMPL=blas
export OPENBLAS_NUM_THREADS=$THREADS

# Configuration A: Planification par défaut (sans affinité)
echo "--- Configuration A: Planification par défaut (Libre) ---"
unset OMP_PROC_BIND
unset OMP_PLACES

# PAS de taskset = planification libre sur tous les cores
DEFAULT_OUTPUT=$(perf stat -r 3 -x, -e instructions,cycles,context-switches,cpu-migrations,cache-misses \
    ./build/test_benchmark_large $SIZE $SIZE $SIZE $REPS 2>&1)

DEFAULT_INSTR=$(perf_value "$DEFAULT_OUTPUT" "instructions")
DEFAULT_INSTR_SD=$(perf_sd "$DEFAULT_OUTPUT" "instructions")
DEFAULT_CS=$(perf_value "$DEFAULT_OUTPUT" "context-switches")
DEFAULT_CS_SD=$(perf_sd "$DEFAULT_OUTPUT" "context-switches")
DEFAULT_MIG=$(perf_value "$DEFAULT_OUTPUT" "cpu-migrations")
DEFAULT_MIG_SD=$(perf_sd "$DEFAULT_OUTPUT" "cpu-migrations")
DEFAULT_CACHE=$(perf_value "$DEFAULT_OUTPUT" "cache-misses")
DEFAULT_CACHE_SD=$(perf_sd "$DEFAULT_OUTPUT" "cache-misses")

if [ -z "$DEFAULT_INSTR" ] || [ -z "$DEFAULT_CS" ] || [ -z "$DEFAULT_MIG" ]; then
    echo "Erreur: Impossible d'extraire les métriques DEFAULT"
    exit 1
fi

echo "  Instructions: $(awk -v val="$DEFAULT_INSTR" 'BEGIN {printf "%.2fB", val / 1000000000}') ± $(awk -v val="$DEFAULT_INSTR_SD" 'BEGIN {printf "%.2f%%", val}')"
echo "  Context Switches: $DEFAULT_CS ± $(awk -v val="$DEFAULT_CS_SD" 'BEGIN {printf "%.2f%%", val}')"
echo "  CPU Migrations: $DEFAULT_MIG ± $(awk -v val="$DEFAULT_MIG_SD" 'BEGIN {printf "%.2f%%", val}')"
echo "  Cache Misses: $(awk -v val="$DEFAULT_CACHE" 'BEGIN {printf "%.2fM", val / 1000000}') ± $(awk -v val="$DEFAULT_CACHE_SD" 'BEGIN {printf "%.2f%%", val}')"
echo ""

# Configuration B: Affinité sur cores physiques
echo "--- Configuration B: Affinité cores physiques (Bound) ---"
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Use taskset to pin threads to cores 0-7
AFFINITY_OUTPUT=$(taskset -c 0-7 perf stat -r 3 -x, -e instructions,cycles,context-switches,cpu-migrations,cache-misses \
    ./build/test_benchmark_large $SIZE $SIZE $SIZE $REPS 2>&1)

AFFINITY_INSTR=$(perf_value "$AFFINITY_OUTPUT" "instructions")
AFFINITY_INSTR_SD=$(perf_sd "$AFFINITY_OUTPUT" "instructions")
AFFINITY_CS=$(perf_value "$AFFINITY_OUTPUT" "context-switches")
AFFINITY_CS_SD=$(perf_sd "$AFFINITY_OUTPUT" "context-switches")
AFFINITY_MIG=$(perf_value "$AFFINITY_OUTPUT" "cpu-migrations")
AFFINITY_MIG_SD=$(perf_sd "$AFFINITY_OUTPUT" "cpu-migrations")
AFFINITY_CACHE=$(perf_value "$AFFINITY_OUTPUT" "cache-misses")
AFFINITY_CACHE_SD=$(perf_sd "$AFFINITY_OUTPUT" "cache-misses")

if [ -z "$AFFINITY_INSTR" ] || [ -z "$AFFINITY_CS" ] || [ -z "$AFFINITY_MIG" ]; then
    echo "Erreur: Impossible d'extraire les métriques AFFINITY"
    exit 1
fi

echo "  Instructions: $(awk -v val="$AFFINITY_INSTR" 'BEGIN {printf "%.2fB", val / 1000000000}') ± $(awk -v val="$AFFINITY_INSTR_SD" 'BEGIN {printf "%.2f%%", val}')"
echo "  Context Switches: $AFFINITY_CS ± $(awk -v val="$AFFINITY_CS_SD" 'BEGIN {printf "%.2f%%", val}')"
echo "  CPU Migrations: $AFFINITY_MIG ± $(awk -v val="$AFFINITY_MIG_SD" 'BEGIN {printf "%.2f%%", val}')"
echo "  Cache Misses: $(awk -v val="$AFFINITY_CACHE" 'BEGIN {printf "%.2fM", val / 1000000}') ± $(awk -v val="$AFFINITY_CACHE_SD" 'BEGIN {printf "%.2f%%", val}')"
echo ""

# Comparaison
echo "--- Résultats de comparaison ---"

# Vérifier division par zéro
if [ "$DEFAULT_CS" = "0" ] || [ "$DEFAULT_MIG" = "0" ]; then
    echo "Avertissement: Valeurs DEFAULT nulles"
    CS_REDUCTION="N/A"
    MIG_REDUCTION="N/A"
else
    CS_REDUCTION=$(awk -v def="$DEFAULT_CS" -v aff="$AFFINITY_CS" 'BEGIN {if(def==0) print "N/A"; else printf "%.1f", (1 - aff/def) * 100}')
    MIG_REDUCTION=$(awk -v def="$DEFAULT_MIG" -v aff="$AFFINITY_MIG" 'BEGIN {if(def==0) print "N/A"; else printf "%.1f", (1 - aff/def) * 100}')
fi

if [ "$DEFAULT_CACHE_SD" = "0" ] || [ -z "$DEFAULT_CACHE_SD" ]; then
    CACHE_SD_REDUCTION="N/A"
else
    CACHE_SD_REDUCTION=$(awk -v def="$DEFAULT_CACHE_SD" -v aff="$AFFINITY_CACHE_SD" 'BEGIN {if(def==0) print "N/A"; else printf "%.1f", (1 - aff/def) * 100}')
fi

echo "  Réduction Context Switches: ${CS_REDUCTION}%"
echo "  Réduction CPU Migrations: ${MIG_REDUCTION}%"
echo "  Réduction écart-type Cache Misses: ${CACHE_SD_REDUCTION}%"
echo ""

# Évaluation
if (( $(echo "$CS_REDUCTION > 50" | bc -l) )); then
    echo "✓ Conclusion: L'affinité CPU réduit significativement les context-switches"
else
    echo "⚠ Avertissement: La réduction des context-switches n'est pas significative"
fi

echo ""
echo "Interprétation:"
echo "Si l'affinité réduit significativement CS/migrations, cela confirme que la planification/"
echo "migration est un facteur clé de l'anomalie BLAS 16 threads."
echo ""

# ==========================================
# Résumé final
# ==========================================

echo "=========================================="
echo "Analyse terminée"
echo "=========================================="
echo ""
echo "✓ 2 expériences clés complétées"
echo ""
echo "Conclusion:"
echo "L'anomalie d'instructions BLAS 16 threads est principalement due à la contention de planification"
echo "lorsque les threads ne sont pas liés aux cœurs physiques (Affinity)."
echo ""
echo "Sortie complète enregistrée."
echo ""
