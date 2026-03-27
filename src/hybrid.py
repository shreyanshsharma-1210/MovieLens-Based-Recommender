import numpy as np

class WeightedHybridRecommender:
    def __init__(self, models, weights):
        """
        models: dict of name -> initialized model instance
        weights: dict of name -> float weight (should sum to 1.0)
        """
        self.models = models
        self.weights = weights
        
    def fit(self, train_data, movies_df=None):
        print("Building Hybrid Model...")
        from content_based import ContentBasedRecommender
        for name, model in self.models.items():
            print(f"  Fitting {name}...")
            if isinstance(model, ContentBasedRecommender):
                model.fit(train_data, movies_df)
            else:
                model.fit(train_data)
        print("Hybrid Model built successfully.")
        
    def predict(self, user_id, item_id):
        final_prediction = 0.0
        total_weight = 0.0
        
        for name, model in self.models.items():
            pred = model.predict(user_id, item_id)
            weight = self.weights.get(name, 0.0)
            
            final_prediction += pred * weight
            total_weight += weight
            
        if total_weight > 0:
            final_prediction /= total_weight
            
        return np.clip(final_prediction, 1.0, 5.0)
