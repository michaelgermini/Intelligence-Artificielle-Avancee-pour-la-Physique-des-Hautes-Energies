# 8.2 Pruning Structuré

---

## Introduction

Le **pruning structuré** supprime des structures entières (filtres, canaux, couches) plutôt que des poids individuels. Cette approche permet une accélération réelle sur hardware standard sans nécessiter d'architectures spécialisées pour la sparsité.

---

## 8.2.1 Filter Pruning

### Principe

Supprime des filtres entiers dans les couches convolutionnelles.

```python
import torch
import torch.nn as nn
import numpy as np

class FilterPruner:
    """
    Pruning de filtres complets dans les couches convolutionnelles
    """
    
    def __init__(self, model):
        self.model = model
        
    def compute_filter_importance(self, criterion='l1'):
        """
        Calcule l'importance de chaque filtre
        
        Critères possibles:
        - 'l1': norme L1 des poids du filtre
        - 'l2': norme L2
        - 'apoz': average percentage of zeros dans les activations
        """
        importance = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                weights = module.weight.data  # (out_channels, in_channels, kH, kW)
                
                if criterion == 'l1':
                    # Norme L1 par filtre
                    imp = weights.abs().sum(dim=(1, 2, 3))
                elif criterion == 'l2':
                    # Norme L2 par filtre
                    imp = (weights ** 2).sum(dim=(1, 2, 3)).sqrt()
                elif criterion == 'apoz':
                    # APoZ: nécessite forward pass sur données
                    # Pour l'instant, utilisons L1
                    imp = weights.abs().sum(dim=(1, 2, 3))
                
                importance[name] = imp
        
        return importance
    
    def prune_filters(self, module_name, n_filters_to_keep, importance):
        """
        Prune une couche conv en gardant les n_filters_to_keep filtres les plus importants
        """
        module = dict(self.model.named_modules())[module_name]
        
        if not isinstance(module, nn.Conv2d):
            raise ValueError(f"{module_name} n'est pas une Conv2d")
        
        # Indices des filtres à garder
        _, indices = torch.topk(importance[module_name], n_filters_to_keep)
        indices = indices.sort()[0]
        
        # Crée une nouvelle couche
        pruned_module = nn.Conv2d(
            module.in_channels,
            n_filters_to_keep,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=module.bias is not None
        )
        
        # Copie les poids et biais
        pruned_module.weight.data = module.weight.data[indices].clone()
        if module.bias is not None:
            pruned_module.bias.data = module.bias.data[indices].clone()
        
        return pruned_module, indices

# Exemple
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 256, 3, padding=1)
)

pruner = FilterPruner(model)
importance = pruner.compute_filter_importance('l1')

print("Importance des filtres:")
for name, imp in importance.items():
    print(f"  {name}: {len(imp)} filtres, importance range: [{imp.min():.2f}, {imp.max():.2f}]")
```

### Pruning Itératif de Filtres

```python
def iterative_filter_pruning(model, train_loader, val_loader, 
                            target_sparsity=0.5, n_iterations=5):
    """
    Pruning itératif de filtres avec fine-tuning
    """
    from copy import deepcopy
    
    current_sparsity = 0
    sparsity_per_iter = target_sparsity / n_iterations
    
    history = {'sparsity': [], 'accuracy': []}
    
    for iteration in range(n_iterations):
        # Calcule l'importance
        pruner = FilterPruner(model)
        importance = pruner.compute_filter_importance('l1')
        
        # Prune chaque couche
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Conv2d):
                current_filters = module.out_channels
                target_filters = int(current_filters * (1 - sparsity_per_iter))
                target_filters = max(1, target_filters)  # Garde au moins 1 filtre
                
                pruned_module, _ = pruner.prune_filters(
                    name, target_filters, importance
                )
                # Remplace dans le modèle (nécessite logique de remplacement)
        
        # Fine-tune
        fine_tune_model(model, train_loader, epochs=3)
        
        # Évalue
        accuracy = evaluate(model, val_loader)
        current_sparsity = compute_model_sparsity(model)
        
        history['sparsity'].append(current_sparsity)
        history['accuracy'].append(accuracy)
        
        print(f"Iteration {iteration+1}: Sparsity={current_sparsity:.1%}, "
              f"Accuracy={accuracy:.2%}")
    
    return model, history
```

---

## 8.2.2 Channel Pruning

### Pruning de Canaux d'Entrée

```python
class ChannelPruner:
    """
    Pruning de canaux d'entrée dans les couches convolutionnelles
    
    Nécessite de propager le pruning aux couches suivantes
    """
    
    def __init__(self, model):
        self.model = model
        
    def compute_channel_importance(self, layer_name, criterion='l1'):
        """
        Calcule l'importance des canaux d'entrée
        """
        module = dict(self.model.named_modules())[layer_name]
        
        if not isinstance(module, nn.Conv2d):
            raise ValueError("Module doit être Conv2d")
        
        weights = module.weight.data  # (out_channels, in_channels, kH, kW)
        
        if criterion == 'l1':
            # Somme sur toutes les sorties et les dimensions spatiales
            importance = weights.abs().sum(dim=(0, 2, 3))
        elif criterion == 'l2':
            importance = (weights ** 2).sum(dim=(0, 2, 3)).sqrt()
        
        return importance
    
    def prune_input_channels(self, module_name, n_channels_to_keep, 
                            channel_indices=None):
        """
        Prune les canaux d'entrée d'une couche
        
        Si channel_indices est None, garde les n_channels_to_keep plus importants
        """
        module = dict(self.model.named_modules())[module_name]
        
        if channel_indices is None:
            importance = self.compute_channel_importance(module_name)
            _, channel_indices = torch.topk(importance, n_channels_to_keep)
            channel_indices = channel_indices.sort()[0]
        
        # Crée nouvelle couche
        pruned_module = nn.Conv2d(
            n_channels_to_keep,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=module.bias is not None
        )
        
        # Copie les poids (sélectionne les canaux d'entrée)
        pruned_module.weight.data = module.weight.data[:, channel_indices].clone()
        if module.bias is not None:
            pruned_module.bias.data = module.bias.data.clone()
        
        return pruned_module, channel_indices
    
    def propagate_pruning(self, pruned_channels, next_layer_name):
        """
        Propage le pruning aux couches suivantes
        
        Si on prune les canaux de sortie d'une couche,
        il faut aussi pruner les canaux d'entrée de la suivante
        """
        # Implémentation simplifiée
        pass

# Exemple avec propagation
def prune_resnet_block(block, target_channels):
    """
    Prune un bloc ResNet de manière cohérente
    
    Les deux conv d'un bloc doivent avoir le même nombre de canaux
    """
    conv1, bn1, relu, conv2, bn2 = block[0], block[1], block[2], block[3], block[4]
    
    # Prune conv1
    importance = conv1.weight.abs().sum(dim=(0, 2, 3))
    _, indices = torch.topk(importance, target_channels)
    
    # Crée conv1 prunée
    new_conv1 = nn.Conv2d(conv1.in_channels, target_channels, conv1.kernel_size)
    new_conv1.weight.data = conv1.weight.data[indices]
    new_bn1 = nn.BatchNorm2d(target_channels)
    
    # Prune conv2 (doit correspondre aux sorties de conv1)
    new_conv2 = nn.Conv2d(target_channels, conv2.out_channels, conv2.kernel_size)
    new_conv2.weight.data = conv2.weight.data[:, indices]
    new_bn2 = nn.BatchNorm2d(conv2.out_channels)
    
    return nn.Sequential(new_conv1, new_bn1, relu, new_conv2, new_bn2)
```

---

## 8.2.3 Layer Pruning

### Suppression de Couches Entières

```python
class LayerPruner:
    """
    Pruning de couches entières dans un réseau
    """
    
    def __init__(self, model):
        self.model = model
        
    def compute_layer_importance(self, train_loader, criterion):
        """
        Calcule l'importance de chaque couche
        
        Méthode: évalue la performance après suppression de chaque couche
        """
        baseline_accuracy = evaluate(self.model, train_loader)
        
        importance = {}
        layers = list(self.model.named_modules())
        
        for name, module in layers:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Sauvegarde originale
                original_state = module.state_dict().copy()
                
                # Remplace par Identity
                if isinstance(module, nn.Linear):
                    replacement = nn.Identity()
                    replacement.out_features = module.out_features
                else:
                    replacement = nn.Identity()
                
                # Évalue
                # (nécessite logique de remplacement dans le modèle)
                # accuracy = evaluate(self.model, train_loader)
                # importance[name] = baseline_accuracy - accuracy
                
                # Restaure
                module.load_state_dict(original_state)
        
        return importance
    
    def remove_layer(self, layer_name):
        """
        Supprime une couche et reconnecte le réseau
        """
        # Logique complexe: nécessite de reconnecter les couches précédentes
        # et suivantes correctement
        pass

def identify_redundant_layers(model, train_loader, threshold=0.01):
    """
    Identifie les couches redondantes (qui peuvent être supprimées
    sans perte significative de performance)
    """
    baseline_acc = evaluate(model, train_loader)
    redundant = []
    
    # Teste chaque couche
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Simule la suppression (remplacement par Identity)
            # Mesure la perte de performance
            # Si < threshold, marque comme redondante
            pass
    
    return redundant
```

---

## Comparaison Filter vs Channel Pruning

```python
def compare_pruning_methods():
    """
    Compare les différentes méthodes de pruning structuré
    """
    methods = {
        'Filter Pruning': {
            'Granularité': 'Filtre complet',
            'Accélération hardware': 'Bonne',
            'Facilité implémentation': 'Facile',
            'Flexibilité': 'Moyenne'
        },
        'Channel Pruning': {
            'Granularité': 'Canal d\'entrée/sortie',
            'Accélération hardware': 'Très bonne',
            'Facilité implémentation': 'Moyenne (propagation)',
            'Flexibilité': 'Bonne'
        },
        'Layer Pruning': {
            'Granularité': 'Couche entière',
            'Accélération hardware': 'Excellente',
            'Facilité implémentation': 'Difficile (reconnexion)',
            'Flexibilité': 'Faible'
        }
    }
    
    print("Comparaison des méthodes de pruning structuré:")
    print(f"{'Méthode':<20} | {'Granularité':<20} | {'Accélération':<15} | {'Facilité':<15}")
    print("-" * 75)
    for name, info in methods.items():
        print(f"{name:<20} | {info['Granularité']:<20} | {info['Accélération hardware']:<15} | {info['Facilité implémentation']:<15}")

compare_pruning_methods()
```

---

## Pruning Structuré pour Transformers

```python
class TransformerHeadPruner:
    """
    Pruning de têtes d'attention dans les Transformers
    """
    
    def compute_head_importance(self, model, train_loader):
        """
        Calcule l'importance de chaque tête d'attention
        """
        importance = {}
        
        for name, module in model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'num_heads'):
                # Méthode: variance des scores d'attention
                # Têtes avec faible variance sont moins importantes
                pass
        
        return importance
    
    def prune_heads(self, model, head_importance, n_heads_to_keep):
        """
        Prune les têtes d'attention les moins importantes
        """
        # Nécessite modification de l'architecture attention
        # pour réduire num_heads dynamiquement
        pass

class TransformerLayerPruner:
    """
    Pruning de couches entières dans un Transformer
    """
    
    def prune_transformer_layers(self, model, n_layers_to_remove):
        """
        Supprime les n dernières couches (ou les moins importantes)
        """
        # Les Transformers sont souvent sur-paramétrés en profondeur
        # Supprimer quelques couches peut être efficace
        pass
```

---

## Exercices

### Exercice 8.2.1
Implémentez un algorithme de filter pruning qui privilégie les filtres produisant des activations proches de zéro.

### Exercice 8.2.2
Créez une fonction qui propage automatiquement le channel pruning à travers un réseau séquentiel.

### Exercice 8.2.3
Comparez filter pruning et channel pruning sur un ResNet. Lequel préserve mieux les performances ?

---

## Points Clés à Retenir

> 📌 **Le pruning structuré donne une accélération réelle sur hardware standard**

> 📌 **Le filter pruning est plus simple mais le channel pruning peut être plus efficace**

> 📌 **La propagation du pruning entre couches est cruciale pour la cohérence**

> 📌 **Le layer pruning donne les plus fortes accélérations mais réduit la flexibilité**

---

*Section suivante : [8.3 Pruning Dynamique et Adaptatif](./08_03_Dynamique.md)*

