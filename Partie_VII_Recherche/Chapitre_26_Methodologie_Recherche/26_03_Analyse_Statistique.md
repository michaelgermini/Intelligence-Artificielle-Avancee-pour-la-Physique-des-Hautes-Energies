# 26.3 Analyse Statistique des Résultats

---

## Introduction

L'**analyse statistique** est essentielle pour tirer des conclusions valides des résultats expérimentaux. Cette section présente les méthodes statistiques pour analyser les résultats de recherche, incluant tests d'hypothèses, intervalles de confiance, et méthodes d'analyse multivariée.

---

## Tests Statistiques de Base

### Tests d'Hypothèses

```python
import numpy as np
from scipy import stats
from typing import Tuple

class StatisticalAnalysis:
    """
    Analyse statistique de résultats expérimentaux
    """
    
    def __init__(self):
        self.significance_level = 0.05
    
    def t_test(self, group1: np.ndarray, group2: np.ndarray, 
               paired: bool = False) -> Dict:
        """
        Test t pour comparer deux groupes
        
        Args:
            group1, group2: Arrays de valeurs
            paired: Si True, test t apparié
        """
        if paired:
            statistic, p_value = stats.ttest_rel(group1, group2)
        else:
            statistic, p_value = stats.ttest_ind(group1, group2)
        
        result = {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'interpretation': self.interpret_p_value(p_value)
        }
        
        return result
    
    def mann_whitney_u(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Test de Mann-Whitney U (non-paramétrique)"""
        statistic, p_value = stats.mannwhitneyu(group1, group2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.significance_level
        }
    
    def interpret_p_value(self, p_value: float) -> str:
        """Interprète p-value"""
        if p_value < 0.001:
            return "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            return "Very significant (p < 0.01)"
        elif p_value < 0.05:
            return "Significant (p < 0.05)"
        else:
            return "Not significant (p >= 0.05)"
    
    def anova(self, groups: List[np.ndarray]) -> Dict:
        """
        ANOVA pour comparer plusieurs groupes
        
        Args:
            groups: Liste d'arrays (un par groupe)
        """
        statistic, p_value = stats.f_oneway(*groups)
        
        # Post-hoc tests si significatif
        post_hoc = None
        if p_value < self.significance_level:
            post_hoc = self.tukey_hsd(groups)
        
        return {
            'f_statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'post_hoc': post_hoc
        }
    
    def tukey_hsd(self, groups: List[np.ndarray]) -> Dict:
        """Test de Tukey HSD pour comparaisons multiples"""
        from scipy.stats import tukey_hsd
        
        result = tukey_hsd(*groups)
        
        return {
            'statistic': result.statistic,
            'pvalue': result.pvalue,
            'confidence_interval': result.confidence_interval()
        }

# Exemple
analyzer = StatisticalAnalysis()

# Comparer deux méthodes
method1_scores = np.array([0.95, 0.94, 0.96, 0.93, 0.95])
method2_scores = np.array([0.92, 0.91, 0.93, 0.90, 0.92])

result = analyzer.t_test(method1_scores, method2_scores)
print(f"t-test result: {result}")
```

---

## Intervalles de Confiance

### Estimation avec Intervalles

```python
class ConfidenceIntervals:
    """
    Calcul d'intervalles de confiance
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def mean_confidence_interval(self, data: np.ndarray) -> Tuple[float, float, float]:
        """
        Intervalle de confiance pour la moyenne
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # ddof=1 pour échantillon
        
        # t-distribution
        t_critical = stats.t.ppf(1 - self.alpha/2, df=n-1)
        
        margin = t_critical * (std / np.sqrt(n))
        
        lower = mean - margin
        upper = mean + margin
        
        return mean, lower, upper
    
    def proportion_confidence_interval(self, successes: int, 
                                      trials: int) -> Tuple[float, float, float]:
        """Intervalle de confiance pour proportion"""
        p = successes / trials
        
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        
        margin = z_critical * np.sqrt(p * (1 - p) / trials)
        
        lower = max(0, p - margin)
        upper = min(1, p + margin)
        
        return p, lower, upper
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                     n_bootstrap: int = 1000,
                                     statistic=np.mean) -> Tuple[float, float, float]:
        """Intervalle de confiance par bootstrap"""
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(sample))
        
        lower = np.percentile(bootstrap_stats, (self.alpha/2) * 100)
        upper = np.percentile(bootstrap_stats, (1 - self.alpha/2) * 100)
        mean_stat = statistic(data)
        
        return mean_stat, lower, upper

# Exemple
ci = ConfidenceIntervals(confidence_level=0.95)

scores = np.array([0.95, 0.94, 0.96, 0.93, 0.95, 0.94, 0.96])
mean, lower, upper = ci.mean_confidence_interval(scores)
print(f"Mean: {mean:.4f}, 95% CI: [{lower:.4f}, {upper:.4f}]")
```

---

## Analyse de Variance

### ANOVA et Extensions

```python
class VarianceAnalysis:
    """
    Analyse de variance
    """
    
    def two_way_anova(self, data: Dict, factor1_levels: List, 
                     factor2_levels: List) -> Dict:
        """
        ANOVA à deux facteurs
        
        Args:
            data: Dict {(level1, level2): array_of_values}
            factor1_levels: Niveaux facteur 1
            factor2_levels: Niveaux facteur 2
        """
        # Préparer données pour statsmodels
        import pandas as pd
        from statsmodels.stats.anova import anova_lm
        from statsmodels.formula.api import ols
        
        # Créer DataFrame
        rows = []
        for (f1, f2), values in data.items():
            for value in values:
                rows.append({'factor1': f1, 'factor2': f2, 'value': value})
        
        df = pd.DataFrame(rows)
        
        # Modèle ANOVA
        model = ols('value ~ C(factor1) + C(factor2) + C(factor1):C(factor2)', 
                   data=df).fit()
        anova_table = anova_lm(model, typ=2)
        
        return {
            'anova_table': anova_table,
            'model_summary': model.summary()
        }
    
    def effect_size(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """
        Calcul taille d'effet (Cohen's d)
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        # Cohen's d
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # Interprétation
        if abs(cohens_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "small"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'cohens_d': cohens_d,
            'interpretation': interpretation
        }

# Exemple
variance_analyzer = VarianceAnalysis()

group1 = np.array([0.95, 0.94, 0.96, 0.93, 0.95])
group2 = np.array([0.92, 0.91, 0.93, 0.90, 0.92])

effect = variance_analyzer.effect_size(group1, group2)
print(f"Effect size: {effect}")
```

---

## Analyse de Corrélation

### Corrélations et Régression

```python
class CorrelationAnalysis:
    """
    Analyse de corrélations
    """
    
    def pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Corrélation de Pearson"""
        r, p_value = stats.pearsonr(x, y)
        
        return {
            'correlation': r,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': self.interpret_correlation(r)
        }
    
    def spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Corrélation de Spearman (non-paramétrique)"""
        rho, p_value = stats.spearmanr(x, y)
        
        return {
            'correlation': rho,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def interpret_correlation(self, r: float) -> str:
        """Interprète corrélation"""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "weak"
        elif abs_r < 0.5:
            return "moderate"
        elif abs_r < 0.7:
            return "strong"
        else:
            return "very strong"
    
    def multiple_regression(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Régression multiple"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        model = LinearRegression()
        model.fit(X, y)
        
        predictions = model.predict(X)
        r2 = r2_score(y, predictions)
        
        return {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'r_squared': r2,
            'model': model
        }

# Exemple
correlation_analyzer = CorrelationAnalysis()

compression_ratios = np.array([2, 4, 8, 10, 16])
accuracies = np.array([0.95, 0.93, 0.90, 0.88, 0.85])

corr_result = correlation_analyzer.pearson_correlation(compression_ratios, accuracies)
print(f"Correlation: {corr_result}")
```

---

## Visualisation Statistique

### Graphiques pour Analyse

```python
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalVisualization:
    """
    Visualisations statistiques
    """
    
    def plot_comparison(self, data: Dict[str, np.ndarray], 
                       ylabel: str = "Value"):
        """Boxplots pour comparaison groupes"""
        fig, ax = plt.subplots()
        
        groups = list(data.keys())
        values = list(data.values())
        
        ax.boxplot(values, labels=groups)
        ax.set_ylabel(ylabel)
        ax.set_title("Comparison of Methods")
        
        return fig, ax
    
    def plot_with_confidence_intervals(self, means: np.ndarray,
                                      lower_bounds: np.ndarray,
                                      upper_bounds: np.ndarray,
                                      x_labels: List[str]):
        """Graphique avec intervalles de confiance"""
        fig, ax = plt.subplots()
        
        x = np.arange(len(means))
        ax.errorbar(x, means, 
                   yerr=[means - lower_bounds, upper_bounds - means],
                   fmt='o', capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Mean Value")
        ax.set_title("Means with 95% Confidence Intervals")
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_correlation_matrix(self, data: np.ndarray, 
                               labels: List[str]):
        """Matrice de corrélation"""
        corr_matrix = np.corrcoef(data.T)
        
        fig, ax = plt.subplots()
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Annoter valeurs
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha='center', va='center')
        
        plt.colorbar(im, ax=ax)
        ax.set_title("Correlation Matrix")
        
        return fig, ax
```

---

## Exercices

### Exercice 26.3.1
Effectuez test t pour comparer deux méthodes et interprétez résultats.

### Exercice 26.3.2
Calculez intervalles de confiance pour moyenne de plusieurs expériences.

### Exercice 26.3.3
Effectuez ANOVA pour comparer plusieurs méthodes et post-hoc tests si nécessaire.

### Exercice 26.3.4
Analysez corrélations entre variables (ex: compression ratio vs accuracy).

---

## Points Clés à Retenir

> 📌 **Tests d'hypothèses permettent conclure sur significativité résultats**

> 📌 **Intervalles de confiance quantifient incertitude estimations**

> 📌 **ANOVA compare plusieurs groupes simultanément**

> 📌 **Taille d'effet (Cohen's d) complète p-value pour interprétation**

> 📌 **Corrélations révèlent relations entre variables**

> 📌 **Visualisations aident interprétation et communication résultats**

---

*Section précédente : [26.2 Design d'Expériences](./26_02_Design_Experiences.md) | Section suivante : [26.4 Reproductibilité](./26_04_Reproductibilite.md)*

