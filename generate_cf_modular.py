import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import dice_ml
from dice_ml.utils import helpers
import pandas as pd

class CounterfactualGenerator:
    def __init__(self, model_path, data_path, num_cfs=2):
        self.num_cfs = num_cfs
        self.model = self.load_model(model_path)
        self.data = self.load_data(data_path)
        self.dice_data = self.prepare_dice_data()
        self.exp = self.create_dice_explainer()

    def load_model(self, model_path):
        return pickle.load(open(model_path, 'rb'))

    def load_data(self, data_path):
        dataset = helpers.load_adult_income_dataset(data_path)
        target = dataset["income"]
        train_dataset, _, _, _ = train_test_split(dataset, target, test_size=0.2, random_state=0, stratify=target)
        return train_dataset

    def prepare_dice_data(self):
        return dice_ml.Data(dataframe=self.data, continuous_features=['age', 'hours_per_week'], outcome_name='income')

    def create_dice_explainer(self):
        return dice_ml.Dice(self.dice_data, dice_ml.Model(model=self.model, backend="sklearn"), method="random")

    def generate_counterfactuals(self, instance):

        # get the counterfactual results 
        counterfactuals_result = self.exp.generate_counterfactuals(instance, total_CFs=self.num_cfs, desired_class="opposite")
        
        #return the dataframe at index 0 that has table of cfs
        return counterfactuals_result.cf_examples_list[0].final_cfs_df

    # Compare the Cf and specify the index of counterfactual for comparision 
    def compare_dfs(self, instance, counterfactuals,index_cf):
        df1 = instance.reset_index(drop=True)
        df2= counterfactuals[index_cf:index_cf+1]
        df2 = df2.iloc[:, :-1].copy().reset_index(drop=True)
        result = df1.compare(df2, align_axis=0)
        return result.reset_index().drop(["level_0", "level_1"], axis=1)

#--------------------------------------------------------------------------------------
def main():
    model_path = "model.pkl"
    data_path = "your_data.csv"
    cfs_generator = CounterfactualGenerator(model_path, data_path)

    x_test = cfs_generator.data.drop('income', axis=1)
    instance = x_test.iloc[0:1]

    cfs = cfs_generator.generate_counterfactuals(instance)
    print("Generated Counterfactuals:")
    print(cfs)

    comparison_result = cfs_generator.compare_dfs(instance, cfs,1)
    print("\nComparison with x_test:")
    print(comparison_result)

if __name__ == "__main__":
    main()