# 📍 Reconstruction de Trajectoires GPS par Interpolation

## 🧭 Description du Projet

Ce projet a pour objectif la **reconstruction d'une trajectoire 2D GPS** à partir de points échantillonnés. Concrètement, on cherche à retrouver une fonction continue $(x(t), y(t))$ à partir d’un ensemble discret de points $(t_i, x_i, y_i)$. Le tout sera évalué à l’aide de différentes **métriques d’erreur** afin d’estimer la qualité de la reconstruction.

Le jeu de données est composé de **50 à 200 points GPS**, éventuellement perturbés par un **bruit gaussien léger**.

---

## 🧪 Objectifs

- Implémenter une interpolation séparée de $x(t)$ et $y(t)$ à partir des points fournis.
- Comparer plusieurs méthodes d'interpolation :
  - Interpolation de **Lagrange**
  - Interpolation de **Newton**
  - **Spline cubique** (recommandé)
- **Tracer** la trajectoire réelle vs la trajectoire reconstruite.
- **Mesurer l’erreur** de reconstruction selon plusieurs métriques :
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
- Présenter l’évolution de l’erreur en fonction du **nombre de points utilisés**.

---

## 📈 Données

- Format : $(t_i, x_i, y_i)$
- Taille : entre **50 et 200 points**
- Possibilité d’ajouter un **bruit gaussien léger** sur les données

---

## 📦 Livrables attendus

- 📊 **Graphiques** : Trajectoire 2D réelle vs reconstruite
- 📋 **Tableau** : Erreurs (MAE, RMSE) pour différents nombres de points
- 📁 Code propre, modulaire et documenté

---

## 🚀 Pistes d'Extension

- Reparamétrage de la trajectoire par **longueur d’arc**
- Ajout de **contraintes de monotonicité** sur $t$
- Amélioration de la robustesse face au bruit

---

## 👥 Règles de Collaboration Git

> ✅ **Une branche par personne obligatoire.**

Chaque membre du projet doit travailler sur **sa propre branche**. Cela permet :
- D'éviter les conflits inutiles
- De garder un historique clair
- De faciliter les revues de code et les fusions

Convention de nommage des branches :  
`prenom/feature` ou `prenom/bugfix`

Exemples :
- `alice/spline-interpolation`
- `bob/erreur-metrics`

---

## 🛠️ Technologies recommandées

- Python 3.x
- Bibliothèques utiles :
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `pandas` (pour les tableaux d’erreur)

---

## ✅ À faire

- [ ] Générer ou importer les données GPS
- [ ] Implémenter les méthodes d’interpolation
- [ ] Implémenter les fonctions de calcul d’erreur
- [ ] Tracer les courbes
- [ ] Créer le tableau comparatif des erreurs
- [ ] (Optionnel) Ajouter du bruit gaussien
- [ ] (Optionnel) Implémenter les extensions

---

## 📌 Auteur·e·s

- Jules RUMEAU - Emilien RESTOUEIX - Anthony ENJALBERT

---

## 📄 Licence

Ce projet est open-source. Vous pouvez le modifier, le redistribuer, ou l'améliorer librement dans le cadre éducatif.

