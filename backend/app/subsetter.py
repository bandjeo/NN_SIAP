import json
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import find
import numpy as np

class Subsetter:
    def __init__(self):
        super().__init__()
        with open("../dataset/layers/layer4.json", "r") as f:
            dataset = json.load(f)

        self.corpus = [' '.join(ingredients) for ingredients in dataset]
        self.corpus_len = len(self.corpus)
        self.vectorizer = CountVectorizer()
        self.X = self.vectorizer.fit_transform(self.corpus)

    def subset(self, ingredients_list):
        n = 25
        ingredients_string = ' '.join(ingredients_list)
        vector = self.vectorizer.transform([ingredients_string])[0]
        scalar_products = self.X.multiply(vector).sum(1)
        knn = np.squeeze(np.asarray(scalar_products.reshape((len(self.corpus))))).argsort()[-n:]
        all_overlapping = {}
        for nearest in knn:
            found_ingredients_indices = find(self.X[nearest])[1]
            for index in found_ingredients_indices:
                ingredient = self.vectorizer.get_feature_names()[index]
                if ingredient in ingredients_list:
                    all_overlapping[ingredient] = all_overlapping.get(ingredient, 0) + 1
        best_ingredients = list(all_overlapping.keys())
        return best_ingredients