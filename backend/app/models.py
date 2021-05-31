
from simpletransformers.seq2seq import Seq2SeqModel
from app.subsetter import Subsetter 
import json

class AbstractModel:
    def __init__(self):
        pass

    def get_instructions(self, ingredient_strings):
        raise Exception('get_instructions feature not implemented.')

    def get_ingredients(self):
        raise Exception('get_ingredients feature not implemented.')


class TransformerModel(AbstractModel):
    def __init__(self):
        super().__init__()
        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": 64,
            "train_batch_size": 16,
            "num_train_epochs": 3,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            # "silent": True,
            "evaluate_generated_text": False,
            "evaluate_during_training": False,
            "evaluate_during_training_verbose": False,
            "use_multiprocessing": False,
            "save_best_model": True,
            "max_length": 200,
            "do_sample": True,
            "top_k": 3,
        }
        self.model = Seq2SeqModel("bert",encoder_decoder_name="app/model",
            args=model_args,
            use_cuda=False,)

        with open('app/ingredients_and_mapper.json') as ingredients_and_mapper_file:
            _ingredients_and_mapper = json.load(ingredients_and_mapper_file)
            self.ingredients = list(dict.fromkeys(_ingredients_and_mapper['frontend']))
            self.mapper = _ingredients_and_mapper['mapper']
            self.small_ingredients = [i for i in self.ingredients if "".join(i.lower().split()) in self.mapper]

        print('initializing subsetter')
        self.subsetter = Subsetter()
        print('initialized subsetter')

    def get_instructions(self, ingredient_strings):
        mapped_ingredient_strings = []
        for ingredient_string in ingredient_strings:
            reformated = "".join(ingredient_string.lower().split())
            if reformated in self.mapper:
                mapped_ingredient_strings.append(self.mapper[reformated])
        ingredients_subset = self.subsetter.subset(mapped_ingredient_strings)
        print('pre: ', mapped_ingredient_strings)
        print('post: ', ingredients_subset)
        return self.model.predict(ingredients_subset)[0]

    def get_ingredients(self):
        return self.small_ingredients

