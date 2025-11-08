---
plantuml:
  server: https://www.plantuml.com/plantuml
---

# Conception détaillée et UML

## Diagramme de classes UML

![](https://www.plantuml.com/plantuml/png/fLLBSjim3Dtp55pQJPG3S3fjckOpcTfnfquEW4YmpOb4Sa0e7pVssJia9PkAdLLN8a3YlGSGeCuz0t7RrAB61KksalHonRBIX2LhQuGeQ_1UtgFnbkHV8UL91GnAJe59lmRRhy3XQApaF2o2Pura_MWSbWIciisjqPni9Z3UMO-ZiTW8eGqZBQ14-QDa_Go1yxeraJknBIoz05MZLzM96FUBiFi8vTDm-rGk2QfPouVdDrqoVt3oq2tdUjbONVLPWbc98qA-lf-TfQZFDdkSp_OZSiSoqGQ19l26gQ352uQY-oolub73m8OThZfIgRnEshXZkKjvue2WGKRouR9gxNbxPolSYet89TcYE4RHYPzEVjyXlrXwIjhh4ECxk0nc6nYXgg8nYgTKdldulpLdIhuKHOxTQmqdKnCzGzrYZXjwv0CF6lO3YgozhasWVmutd3TTxzlqZ3E75CksfxRk_isXaZFtIviONjh6jHmdPY-0SAaz8rqSOWz1Y-BzMypim9_SKiakcNow9jRxutFb22iL8qyb0osHapLEdztGJETN0QIpVfBr_iemMVSMtGE2sF9zmE7EGR20Hyy49SkPOzteTJSPsVL_l9cQb5pK5ZY-G8mklw3m_jmGv4NinltcSiXGDBP1mstp_JEuZHW4nVBlpDrKtDtExo-1mICdvRNDwAKyCPOQgFKQ3vicn4qiLBsaAk6e0Mfy583BrLolKKbjebA3htmHl06LMrVZjdy5qpSCIeOVccI0sZzy1DhiIr9XhR99ZtCURj6eGIC5l-rBeRAgT4aaKfKbsFq1jBboDedkwuDXH25DnHIPTEd5HlIwekwzdjmz49HlxENatapBEPeg_Fh-0G00)

## Description de la conception

- `Tensor`/`Matrix` gèrent toutes les opérations fondamentales d’algèbre linéaire et offrent un conteneur de données unifié pour la propagation avant et la différentiation automatique.  
- La couche d’abstraction `ActivationFunction` masque les détails des différentes activations. `LinearLayer`, combinée à l’activation, expose une interface `forward` unique, ce qui facilite l’empilement dans `MLPNetwork`.  
- L’autodifférentiation est représentée par `Node`, entité différentiable du graphe de calcul ; `OperationNode` encapsule les opérateurs spécifiques (produit matriciel, opérations élémentaires, agrégations) et enregistre `backwardFn` pour la rétropropagation.  
- `LossFunction` / `Optimizer` adoptent le patron de stratégie, permettant de substituer pertes et algorithmes d’optimisation sans modifier `Trainer`.  
- `DataLoader` gère le chargement des données MNIST, la normalisation et la génération des mini-batchs, tandis que `Trainer` orchestre les boucles de lot, `forward → loss → backward → optimizer.step()` ainsi que l’enregistrement des métriques d’évaluation.  
- Cette architecture est alignée sur les phases définies dans `Docs/description_de_la_demande.md` et autorise des extensions futures (nouvelles activations, pertes, optimiseurs, etc.) sans toucher au flux d’entraînement principal.

