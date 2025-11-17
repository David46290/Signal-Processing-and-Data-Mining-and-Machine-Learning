import numpy as np

class stackingModel:
    def __init__(self, model_lst, name_lst, final_estimator, final_estimator_name):
        """
        Parameters
        ----------
        model_lst : list
            [model1, model2. model3]; all unfit models in first layer
            
        name_lst : list of str
            name of model in first layer
            
        final_estimator : a class that is a machine learning model type
        """
        self.model_lst = model_lst
        self.name_lst = name_lst
        
        self.model_dict = {name_lst[0]:model_lst[0]}
        if len(model_lst) > 1 and len(name_lst) > 1:
            for model_idx, model in enumerate(model_lst[1:]):
                self.model_dict.update({name_lst[model_idx+1]:model})
        self.final_estimator = final_estimator
        self.final_estimator_name = final_estimator_name
      
    def input_final_estimator(self, x):
        final_input = []
        for model_idx, model_name in enumerate(self.name_lst):
            predicted_y = self.model_dict[model_name].predict(x)
            # print(f"Score of {model_name} (1st layer):", self.model_dict[model_name].score(x, y))
            # if len(predicted_y.shape) > 1: # if y is like (num_sample, 1) not (numsample)
            #     predicted_y = predicted_y.reshape(-1)
            final_input.append(predicted_y)
        final_input = np.array(final_input).T      
        return final_input
    
    def fit(self, x, y):
        # training first layer
        
        for model_idx, model_name in enumerate(self.name_lst):
            self.model_dict[model_name].fit(x, y)  
        # creating input_final_estimator for final_estimator
        input_final_estimator = self.input_final_estimator(x)
        # training final layer
        self.final_estimator.fit(input_final_estimator, y)
        # print(f"Score of {self.final_estimator_name} (final layer):", self.final_estimator.score(input_final_estimator, y))
        
    def predict(self, x):
        input_final_estimator = self.input_final_estimator(x)
        y_pred = self.final_estimator.predict(input_final_estimator)
        return y_pred
            